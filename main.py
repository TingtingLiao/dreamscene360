import os
import sys 
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg
import kiui
import torch
import torch.nn.functional as F
import random  
import imageio
from PIL import Image
from scipy.spatial.transform import Rotation as R  
from txt2panoimg import Text2360PanoramaImagePipeline
from img2panoimg import Image2360PanoramaImagePipeline 
from diffusers.utils import load_image

python_src_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Adding '{python_src_dir}' to sys.path") 
sys.path.append(python_src_dir + "/submodules/360monodepth/code/python/src/") 

from gs_renderer import Renderer, MiniCam 
from utils import cam_utils
from utils.loss_utils import l1_loss, ssim
from utils import depthmap_align 
from utils.Equirec2Perspec import Equirectangular

from utility import blending, image_io, depthmap_utils, serialization, pointcloud_utils
from utility.logger import Logger
from utility.projection_icosahedron import erp2ico_image
 

log = Logger(__name__)
log.logger.propagate = True 
 

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        
        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda") 
        self.cam = cam_utils.OrbitCamera(opt.W, opt.H, r=0, fovy=opt.fovy)
        self.auto_rotate = True
    
        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree, white_background=self.opt.white_background)
        self.bg_color = self.renderer.bg_color
        
        self.gaussain_scale_factor = 1
        self.fovy = self.opt.fovy 
 
        # input text
        self.prompt = "" if self.opt.prompt is None else self.opt.prompt

        # load panorama image generation model 
        model_path ="~/.cache/huggingface/hub/models--archerfmy0831--sd-t2i-360panoimage/snapshots"
        model_path = os.path.expanduser(model_path)
        model_path = os.path.join(model_path, os.listdir(model_path)[0])
        self.txt2panoimg = Text2360PanoramaImagePipeline(model_path, torch_dtype=torch.float16)
        self.img2panoimg = Image2360PanoramaImagePipeline(model_path, torch_dtype=torch.float16)

        # training stuff
        self.training = False 
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # self.load_models() 
        self.opt.save_path = f"{self.prompt}" # _{time.strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(self.opt.outdir, self.opt.save_path)
        os.makedirs(self.log_dir, exist_ok=True)

        self.subimages_dir = os.path.join(self.log_dir, "subimages/")
        self.depth_estimate_dir = os.path.join(self.log_dir, "depth_estimate/")
        self.depth_align_dir = os.path.join(self.log_dir, "depth_align/")

        self.erp_image = None
        self.subimg_rgb_list = [] 
        self.subimg_gnomo_xy = []  # to convert the perspective image to ERP image
        # stage 2: depth estimation
        self.depthmap_persp_list = []
        self.depthmap_erp_list = []  # the depth map in ERP image space
        self.dispmap_erp_list = []
        self.dispmap_persp_list = []
        # stage 3: depth alignment
        self.dispmap_aligned_list = [] 
        self.subimg_cam_list = [] 
        
        if self.opt.load is not None:
            try:
                self.renderer.gaussians.load_ply(self.opt.load)
            except:
                self.renderer.initialize(self.opt.load)   
        else:
            self.renderer.initialize(num_pts=self.opt.num_pts)       
 
        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()
  
    @torch.no_grad()
    def generate_panorama_image(self):   
        if self.opt.input is None:   
            input = {'prompt': self.prompt, 'upscale': self.opt.upscale} 
            output = self.txt2panoimg(input) 
        else:
            image = load_image(self.opt.input).resize((512, 512))
            mask = load_image("./data/i2p-mask.jpg") 
            input = {'prompt': self.prompt, 'upscale': self.opt.upscale, 'image':image, 'mask': mask} 
            output = self.img2panoimg(input)  
        output.save(f"{self.log_dir}/panorama_image.png")
         
    @torch.no_grad()
    def panorama_to_tangent_images(self): 
        # load panorama image 
        self.erp_image = cv2.cvtColor(cv2.imread(f"{self.log_dir}/panorama_image.png"), cv2.COLOR_BGR2RGB)
        self.erp_image = cv2.resize(self.erp_image, (self.opt.pano_width, self.opt.pano_height))

        # projection to tangent images
        self.subimg_rgb_list, _, points_gnomocoord = erp2ico_image(
            self.erp_image, self.opt.tangent_img_width, 
            padding_size=self.opt.subimage_padding_size, 
            full_face_image=True
            )
        self.subimg_gnomo_xy = points_gnomocoord[1]
        
        if self.opt.debug:
            image_io.subimage_save_ico(self.subimg_rgb_list, f"{self.log_dir}/subimages_vis.png")
            os.makedirs(self.subimages_dir, exist_ok=True)
            for index, img in enumerate(self.subimg_rgb_list):
                Image.fromarray(img.astype(np.uint8)).save(f"{self.subimages_dir}/{index:03d}.png")
            print("Output subimages to {}.".format(self.log_dir)) 
    
    @torch.no_grad()
    def depth_estimation(self): 
        # estimate disparity map
        self.dispmap_persp_list = depthmap_utils.run_persp_monodepth(self.subimg_rgb_list, self.opt.persp_monodepth)
        
        # convert disparity map to depth map 
        for dispmap_persp in self.dispmap_persp_list:
            depthmap_persp = depthmap_utils.disparity2depth(dispmap_persp)
            depthmap_erp = depthmap_utils.subdepthmap_tang2erp(depthmap_persp, self.subimg_gnomo_xy)
            dispmap_erp = depthmap_utils.depth2disparity(depthmap_erp).astype(np.float32)
            self.depthmap_persp_list.append(depthmap_persp)
            self.depthmap_erp_list.append(depthmap_erp) 
            self.dispmap_erp_list.append(dispmap_erp)
 
        if self.opt.debug:
            os.makedirs(self.depth_estimate_dir, exist_ok=True)
            depthmap_utils.depth_ico_visual_save(self.dispmap_persp_list, f"{self.depth_estimate_dir}/dispmap_persp.png") 
            depthmap_utils.depth_ico_visual_save(self.depthmap_persp_list, f"{self.depth_estimate_dir}/depthmap_persp.png") 
            depthmap_utils.depth_ico_visual_save(self.depthmap_erp_list, f"{self.depth_estimate_dir}/depthmap.png") 
            depthmap_utils.depth_ico_visual_save(self.dispmap_erp_list, f"{self.depth_estimate_dir}/dispmap.png")
            for i in range(len(self.subimg_rgb_list)):
                depthmap_utils.write_pfm(f"{self.depth_estimate_dir}/depthmap_{i:03d}.pfm", self.depthmap_erp_list[i], scale=1) 
                depthmap_utils.write_pfm(f"{self.depth_estimate_dir}/depthmap_perspective{i:03d}.pfm", self.depthmap_persp_list[i], scale=1)
 
    @torch.no_grad()
    def depth_alignment(self):
        # from utils import depthmap_align
        os.makedirs(self.depth_align_dir, exist_ok=True) 
        depthmap_aligner = depthmap_align.DepthmapAlign(self.opt, self.depth_align_dir, self.subimg_rgb_list, debug=True)
             
        subimage_available_list = list(range(len(self.dispmap_erp_list)))
        self.dispmap_aligned_list, coeffs_scale, coeffs_offset, self.subimg_cam_list = \
            depthmap_aligner.align_multi_res(self.erp_image, self.dispmap_erp_list, self.opt.subimage_padding_size, subimage_available_list)
         
        if self.opt.debug:   
            serialization.subimage_alignment_params(f"{self.depth_align_dir}/disp_coeff.json", coeffs_scale, coeffs_offset, subimage_available_list)
            depthmap_utils.depth_ico_visual_save(coeffs_scale, f"{self.depth_align_dir}/scale.png", subimage_available_list)
            depthmap_utils.depth_ico_visual_save(coeffs_offset, f"{self.depth_align_dir}/offset.png", subimage_available_list)
            serialization.save_cam_params(f"{self.depth_align_dir}/camera_all.json",subimage_available_list, self.subimg_cam_list)
            depthmap_utils.depth_ico_visual_save(self.dispmap_aligned_list, f"{self.depth_align_dir}/dispmap_aligned.png")
 
        blend_it = blending.BlendIt(opt.subimage_padding_size, len(subimage_available_list), self.opt.blending_method)
        blend_it.fidelity_weight = 0.1

        erp_image_height = self.erp_image.shape[0]
        blend_it.tangent_images_coordinates(erp_image_height, self.dispmap_aligned_list[0].shape)
        blend_it.erp_blendweights(self.subimg_cam_list, erp_image_height, self.dispmap_aligned_list[0].shape)
        blend_it.compute_linear_system_matrices(erp_image_height, erp_image_height * 2, blend_it.frustum_blendweights)

        erp_dispmap_blend = blend_it.blend(self.dispmap_aligned_list, erp_image_height)
        blending_method = 'poisson' if self.opt.blending_method == 'all' else self.opt.blending_method
        erp_dispmap_blend_save = erp_dispmap_blend[blending_method]  
        pointcloud_utils.depthmap2pointcloud_erp(erp_dispmap_blend_save, self.erp_image, f"{self.log_dir}/pointcloud.ply")
 
        depthmap_utils.write_pfm(f"{self.depth_align_dir}/depth_blending.pfm", erp_dispmap_blend_save.astype(np.float32), scale=1) 
        depthmap_utils.depth_visual_save(erp_dispmap_blend_save, f"{self.log_dir}/depth_blending_vis.png")
  
    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        self.last_seed = seed

    def prepare_train(self): 
        self.generate_panorama_image()
        self.panorama_to_tangent_images()
        self.depth_estimation()
        self.depth_alignment() 
        # exit()
        self.step = 0 
        self.Equirectangular = Equirectangular(f"{self.log_dir}/panorama_image.png")  
        self.renderer.initialize(f"{self.log_dir}/pointcloud.ply")  
        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
       
    def sample_from_equirectangular(self, image_height=512, image_width=512): 
        theta = random.randint(0, 360)
        phi = random.randint(self.opt.min_ver, self.opt.max_ver)  
        img = self.Equirectangular.GetPerspective(self.fovy, -(theta+180), -phi, image_height, image_width) 
        # cv2.imwrite("perspective_image.jpg", img)
        img = torch.as_tensor(img[..., ::-1].copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0
        img = img.to(self.device)
        rot = R.from_euler("zyx", [0, theta, phi], degrees=True).as_matrix() 
        w2c = np.eye(4)
        w2c[:3, :3] = rot
        w2c = w2c.astype(np.float32) 
        camera = MiniCam(
                w2c,
                image_height,
                image_width,
                np.deg2rad(self.fovy),
                np.deg2rad(self.fovy),
                0.1,
                100,
            )
        return img, camera 
     
    def get_known_view_loss(self):  
        # sample image and camera from equirectangular
        gt_image, cam = self.sample_from_equirectangular()

        # render image
        gs_out = self.renderer.render(cam) 
        image, rend_normal, surf_normal = gs_out["image"], gs_out["rend_normal"], gs_out["surf_normal"]
        
        # image loss 
        loss = (1.0 - self.opt.lambda_dssim) * l1_loss(image, gt_image) + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # distortion loss  
        if self.step > 3000:
            loss += self.opt.lambda_dist * gs_out["rend_dist"].mean()
 
        # normal loss   
        if self.step > 7000 and self.opt.lambda_normal > 0:  
            loss += self.opt.lambda_normal * (1 - (rend_normal * surf_normal).sum(dim=0)).mean()    

        # save visualization
        if self.opt.debug and self.step % 100 == 0:
            vis_image = torch.cat([image, gt_image, rend_normal* 0.5 + 0.5, surf_normal* 0.5 + 0.5], dim=2).permute(1, 2, 0)  
            os.makedirs(f'{self.log_dir}/vis/', exist_ok=True)
            cv2.imwrite(f'{self.log_dir}/vis/step_{self.step}.png', (vis_image.detach().cpu().numpy()[..., ::-1] * 255).astype(np.uint8))
 
        return loss, gs_out

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
 
        self.step += 1
          
        self.renderer.gaussians.update_learning_rate(self.step)

        if self.step % 1000 == 0:
            self.renderer.gaussians.oneupSHdegree()
 
        loss, gs_out = self.get_known_view_loss()  
        loss.backward()
        ender.record()
        
        with torch.no_grad():
            # Densification
            if self.step < self.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = gs_out["viewspace_points"], gs_out["visibility_filter"], gs_out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step >= self.opt.density_start_iter and self.step % self.opt.densification_interval == 0:
                    # size_threshold = 20 if self.step > opt.opacity_reset_interval else None
                    self.renderer.gaussians.densify_and_prune(
                        self.opt.densify_grad_threshold, 
                        min_opacity=self.opt.densify_min_opacity, 
                        extent=self.opt.densify_extent,  
                        max_screen_size=None, 
                        )
        
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()
            
            self.renderer.gaussians.optimizer.step()
            self.renderer.gaussians.optimizer.zero_grad(set_to_none=True)
       
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if self.auto_rotate:
            self.cam.orbit(10, 0)

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)
 
            buffer_image = out[self.mode] if self.mode in ["image", "alpha"] else out["surf_"+self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)
            
            if self.mode == 'normal':
                buffer_image = buffer_image * 0.5 + 0.5 

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )
  
            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!
  
    @torch.no_grad()
    def save_model(self): 
        path = os.path.join(self.log_dir, f'{self.opt.save_path}_gs.ply')
        self.renderer.gaussians.save_ply(path)
        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )
 
                # prompt stuff            
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )
 
                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model()

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)
 
                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")
 
                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                # auto rotate camera 
                with dpg.group(horizontal=True):
                    def callback_toggle_auto_rotate(sender, app_data):
                            self.auto_rotate = not self.auto_rotate
                            self.need_update = True
                    dpg.add_checkbox(
                        label="auto rotate",
                        default_value=self.auto_rotate,
                        callback=callback_toggle_auto_rotate,
                    )
                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha", "normal"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=self.fovy,
                    callback=callback_set_fovy,
                )

                def callback_set_gaussain_scale(sender, app_data):
                    self.gaussain_scale_factor = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="gaussain scale",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.gaussain_scale_factor,
                    callback=callback_set_gaussain_scale,
                )

        ### register camera handler 
        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="dreamscene360",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    @torch.no_grad()
    def render_360_video(self, num_cameras=60, render_res=512): 
        log_dir = os.path.join(self.opt.outdir, self.opt.save_path)  
        os.makedirs(log_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')   
        
        out_video = cv2.VideoWriter(f'{log_dir}/{self.prompt}.mp4', fourcc, num_cameras / 10, (render_res*3, render_res))
         
        yaws = torch.linspace(0, 360, num_cameras)  

        print(f"[INFO] rendering 360 video...")
        # for xaw in xaws:   
        for yaw in yaws:   
            rot = R.from_euler("zyx", [0, np.deg2rad(yaw), 0], degrees=False).as_matrix()  
            # rot = R.from_euler("zyx", [0, 0, np.deg2rad(xaw)], degrees=False).as_matrix()  
            w2c = np.eye(4)
            w2c[:3, :3] = rot
            w2c = w2c.astype(np.float32)
             
            # pose = cam_utils.orbit_camera(0, yaw, self.opt.radius)
            # pose = np.linalg.inv(pose) 
            cur_cam = MiniCam(
                w2c, 
                render_res, 
                render_res, 
                np.deg2rad(self.fovy), 
                np.deg2rad(self.fovy),  
                0.01, 100 
            )
            out = self.renderer.render(cur_cam)

            image = out["image"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255   
            rand_normal = (out["rend_normal"] * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy() * 255 
            surf_normal = (out["surf_normal"] * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy() * 255
            
            alpha = out["alpha"].permute(1, 2, 0).cpu().numpy()  
            image = image * alpha + (1 - alpha) * 255
            rand_normal = rand_normal * alpha + (1 - alpha) * 255
            surf_normal = surf_normal * alpha + (1 - alpha) * 255
             
            image = np.hstack([image, rand_normal, surf_normal])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out_video.write(image.astype(np.uint8))
 
        out_video.release() 
        print(f"[INFO] 360 video saved to {log_dir}.")
  
    # no gui mode
    def train(self, iters=100):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step() 
            print(f"[INFO] training done!")

        self.render_360_video()
        self.save_model()
          

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="path to the yaml config file") 
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras)) 
    gui = GUI(opt)
    
    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters)
 
 