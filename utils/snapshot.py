import numpy as np
import os
from dataclasses import dataclass
import trimesh
import imgui
import imageio
import glob
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pathlib import Path

from aitviewer.configuration import CONFIG as C
from aitviewer.scene.node import Node
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer
from aitviewer.utils import get_video_paths, video_to_gif

from utils.mesh import normalize

    


@dataclass
class Snapshot:
    step: int
    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray   # (F, 3) or (F, 4)


class SnapshotViewer(Viewer):
    snap_id: int=0 

    def __init__(self, snapshots: list[Snapshot], **kwargs):
        super().__init__(**kwargs)
        self._snapshots = snapshots
        self.n_snapshots = len(snapshots)
        self.names = [f"snapshot_{i}" for i in range(self.n_snapshots)]
        # self.base_name = kwargs.get('base_name', 'snapshot')
        self.output_dir = kwargs.get('output_dir', None)

        self.color=(0.275966, 0.792482, 0.8, 0.9)

        background_mesh = trimesh.load('/media/box/Elements/Exp/HACK-Model/background.obj')
        background_mesh = Meshes(
            vertices=np.array(background_mesh.vertices), faces=np.array(background_mesh.faces),
            name='background', position=[0.0, 0.0, 0.0],
            color=(1.0, 1.0, 1.0, 1.0), flat_shading=True,
        )
        self.scene.add(background_mesh)
        
        self.scene.origin.enabled = False
        self.scene.floor.enabled = False

        # by default the 1st snapshot is the parent mesh
        p_mesh = Meshes(
            vertices=np.zeros((1,3)), faces=np.zeros((1,3), dtype=np.int32),
            name='Snapshots mesh', position=[0.2, 0.0, 0.0],
            flat_shading=True, 
            draw_edges=False, 
            draw_outline=False,
        )
        p_mesh.norm_coloring = True
        self.scene.add(p_mesh)

        self.scene.get_node_by_name('Snapshots mesh').enabled = True

        # ids = np.hstack([np.array([0]), np.array([self.n_snapshots-1]), np.arange(1,self.n_snapshots-1)])
        for i in range(len(snapshots)):
            mesh = Meshes(
                vertices=snapshots[i].vertices, faces=snapshots[i].faces,
                name=f"snapshot_{i}", position=[0.0, 0.0, 0.0],
                color=self.color, flat_shading=True, draw_edges=False, draw_outline=False,
            )
            mesh.norm_coloring = True
            p_mesh.add(mesh)
            if i != 0:
                self.scene.get_node_by_name('Snapshots mesh').nodes[i].enabled = False

    def gui_scene(self,):
        super().gui_scene()
        '''
        add your scene gui here
        '''
        imgui.set_next_window_position(self.window_size[0] * 0.4, 60, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(self.window_size[0] * 0.6, self.window_size[1] * 0.4, imgui.FIRST_USE_EVER)
        imgui.push_font(self.custom_font)

        imgui.begin("Snapshots", None)
        # imgui.text(f"{"\u0088"}")
        id_changed, self.snap_id = imgui.slider_int(
            label="snap id", value=self.snap_id, min_value=0, max_value=self.n_snapshots-1,
        )
        if id_changed:
            self.scene.get_node_by_name('Snapshots mesh').nodes[self.snap_id]._enabled = True
            # set the mesh to be visible
            invisible_ids = np.delete(np.arange(self.n_snapshots), self.snap_id)
            for i in invisible_ids:
                self.scene.get_node_by_name('Snapshots mesh').nodes[i]._enabled = False

        imgui.pop_font()
        imgui.end()

    def gui_playback(self, ):
        '''
        playback not used
        '''
        pass

    def gui_menu(self):
        clicked_export_video = False
        clicked_export_gif = False
        clicked_screenshot = False
        clicked_export_usd = False
        
        imgui.push_font(self.custom_font)
            
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_quit, selected_quit = imgui.menu_item("Quit", "Cmd+Q", False, True)
                if clicked_quit:
                    exit(1)

                clicked_export_video, selected_export_video = imgui.menu_item("Save as video..", None, False, True)

                clicked_export_gif, selected_export_gif = imgui.menu_item("Save as gif..", None, False, True)

                clicked_screenshot, selected_screenshot = imgui.menu_item(
                    "Screenshot",
                    self._shortcut_names[self._screenshot_key],
                    False,
                    True,
                )

                clicked_export_usd, _ = imgui.menu_item("Save as USD", None, False, True)
                imgui.end_menu()

            if imgui.begin_menu("View", True):
                if imgui.begin_menu("Light modes"):
                    _, default = imgui.menu_item("Default", None, self.scene.light_mode == "default")
                    if default:
                        self.scene.light_mode = "default"

                    _, dark = imgui.menu_item(
                        "Dark",
                        self._shortcut_names[self._dark_mode_key],
                        self.scene.light_mode == "dark",
                    )
                    if dark:
                        self.scene.light_mode = "dark"

                    _, diffuse = imgui.menu_item("Diffuse", None, self.scene.light_mode == "diffuse")
                    if diffuse:
                        self.scene.light_mode = "diffuse"
                    imgui.end_menu()

                _, self.shadows_enabled = imgui.menu_item(
                    "Render Shadows",
                    self._shortcut_names[self._shadow_key],
                    self.shadows_enabled,
                    True,
                )

                _, self.lock_selection = imgui.menu_item(
                    "Lock selection",
                    self._shortcut_names[self._lock_selection_key],
                    self.lock_selection,
                    True,
                )

                if imgui.begin_menu("Viewports"):

                    def menu_entry(text, name):
                        if imgui.menu_item(text, None, self.viewport_mode == name)[1]:
                            self.viewport_mode = name

                    menu_entry("Single", "single")
                    menu_entry("Split vertical", "split_v")
                    menu_entry("Split horizontal", "split_h")
                    menu_entry("Split both", "split_vh")
                    imgui.separator()
                    if imgui.menu_item("Ortho grid +", None)[1]:
                        self.set_ortho_grid_viewports()
                    if imgui.menu_item("Ortho grid -", None)[1]:
                        self.set_ortho_grid_viewports((True, True, True))

                    imgui.end_menu()

                _, self.render_gui = imgui.menu_item("Render GUI", None, self.render_gui, True)

                imgui.end_menu()

            if imgui.begin_menu("Camera", True):
                _, self.scene.camera_target.enabled = imgui.menu_item(
                    "Show camera target",
                    self._shortcut_names[self._show_camera_target_key],
                    self.scene.camera_target.enabled,
                    True,
                )

                _, self.scene.trackball.enabled = imgui.menu_item(
                    "Show camera trackball",
                    self._shortcut_names[self._show_camera_trackball_key],
                    self.scene.trackball.enabled,
                    True,
                )

                clicked, _ = imgui.menu_item(
                    "Center view on selection",
                    self._shortcut_names[self._center_view_on_selection_key],
                    False,
                    isinstance(self.scene.selected_object, Node),
                )
                if clicked:
                    self.center_view_on_selection()

                is_ortho = False if self.viewports[0].using_temp_camera else self.scene.camera.is_ortho
                _, is_ortho = imgui.menu_item(
                    "Orthographic Camera",
                    self._shortcut_names[self._orthographic_camera_key],
                    is_ortho,
                    True,
                )

                if is_ortho and self.viewports[0].using_temp_camera:
                    self.reset_camera()
                self.scene.camera.is_ortho = is_ortho

                if imgui.begin_menu("Control modes", enabled=not self.viewports[0].using_temp_camera):

                    def mode(name, mode):
                        selected = imgui.menu_item(name, None, self.scene.camera.control_mode == mode)[1]
                        if selected:
                            self.reset_camera()
                            self.scene.camera.control_mode = mode

                    mode("Turntable", "turntable")
                    mode("Trackball", "trackball")
                    mode("First Person", "first_person")
                    imgui.end_menu()

                clicked_save_cam, selected_save_cam = imgui.menu_item(
                    "Save Camera", self._shortcut_names[self._save_cam_key], False, True
                )
                if clicked_save_cam:
                    self.reset_camera()
                    self.scene.camera.save_cam()

                clicked_load_cam, selected_load_cam = imgui.menu_item(
                    "Load Camera", self._shortcut_names[self._load_cam_key], False, True
                )
                if clicked_load_cam:
                    self.reset_camera()
                    self.scene.camera.load_cam()

                imgui.end_menu()

            if imgui.begin_menu("Mode", True):
                for id, mode in self.modes.items():
                    mode_clicked, _ = imgui.menu_item(mode["title"], mode["shortcut"], id == self.selected_mode, True)
                    if mode_clicked:
                        self.selected_mode = id

                imgui.end_menu()

            if imgui.begin_menu("Help", True):
                clicked, self._show_shortcuts_window = imgui.menu_item(
                    "Keyboard shortcuts", None, self._show_shortcuts_window
                )
                imgui.end_menu()

            if imgui.begin_menu("Debug", True):
                _, self.visualize = imgui.menu_item(
                    "Visualize debug texture",
                    self._shortcut_names[self._visualize_key],
                    self.visualize,
                    True,
                )

                imgui.end_menu()

            if self.server is not None:
                if imgui.begin_menu("Server", True):
                    imgui.text("Connected clients:")
                    imgui.separator()
                    for c in self.server.connections.keys():
                        imgui.text(f"{c[0]}:{c[1]}")
                    imgui.end_menu()

            imgui.end_main_menu_bar()

        if clicked_export_video:
            imgui.open_popup("Export Video")
            self.export_fps = self.playback_fps
            self.toggle_animation(False)
        
        if clicked_export_gif:
            imgui.open_popup("Export Gif")
            self.toggle_animation(False)

        if clicked_screenshot:
            self._screenshot_popup_just_opened = True

        if clicked_export_usd:
            self._export_usd_popup_just_opened = True

        self.gui_export_video()
        self.gui_export_gif()
        self.gui_screenshot()
        self.gui_export_usd()

        imgui.pop_font()


    def gui_export_gif(self):
        imgui.set_next_window_size(570, 0)
        
        if imgui.begin_popup_modal(
            "Export Gif", flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE 
        )[0]:
            if self.scene.n_frames == 1:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.2)
                self.export_animation = False

            if self.scene.n_frames == 1:
                imgui.pop_style_var()
                self.export_animation = False

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Output settings.
            imgui.text("Output")

            imgui.spacing()
            imgui.text("Format:")
            imgui.same_line()

            if imgui.radio_button("GIF", self.export_format == "gif"):
                # true if clicked
                self.export_format = "gif"
            imgui.same_line(spacing=15)
            if imgui.radio_button("WEBM", self.export_format == "webm"):
                self.export_format = "webm"

            imgui.spacing()
            if self.export_format == "gif":
                # imgui.push_style_var(imgui.STYLE_ALPHA, 0.2)
                self.export_transparent = True

            _, self.export_transparent = imgui.checkbox("Transparent background", True)

            # if self.export_format != "gif":
            #     imgui.pop_style_var()
            #     self.export_transparent = False

            # if imgui.is_item_hovered():
            #     imgui.begin_tooltip()
            #     imgui.text("Available only for WEBM format")
            #     imgui.end_tooltip()

            if self.export_format == "gif":
                max_output_fps = 60.0
            else:
                max_output_fps = 120.0

            self.export_fps = min(self.export_fps, max_output_fps)
            imgui.text(
                f"Duration: [{self.n_snapshots/self.export_fps:.1f}s]"
            )
            fps_changed, self.export_fps = imgui.drag_float(
                label="fps",
                value=self.export_fps,
                # 1.0,
                min_value=2.0,
                max_value=max_output_fps,
                format="%.1f",
            )

            imgui.spacing()
            imgui.spacing()
            imgui.text(
                f"Resolution: [{int(self.window_size[0] * self.export_scale_factor)}x{int(self.window_size[1] * self.export_scale_factor)}]"
            )
            _, self.export_scale_factor = imgui.drag_float(
                label="Scale",
                value=self.export_scale_factor,
                min_value=0.01,
                max_value=1.0,
                change_speed=0.005,
                format="%.2f",
            )

            # Draw a cancel and exit button on the same line using the available space
            button_width = (imgui.get_content_region_available()[0] - imgui.get_style().item_spacing[0]) * 0.5

            # Style the cancel with a grey color
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.5, 0.5, 0.5, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.6, 0.6, 0.6, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.7, 0.7, 0.7, 1.0)

            if imgui.button("Cancel", width=button_width):
                imgui.close_current_popup()

            imgui.pop_style_color()
            imgui.pop_style_color()
            imgui.pop_style_color()

            imgui.same_line()
            if imgui.button("Export", button_width):
                imgui.close_current_popup()
                output_path = os.path.join(
                        self.output_dir,
                        f"animation.{self.export_format}",
                    )
                self.export_gif(
                    output_path=output_path, output_fps=self.export_fps,
                    scale_factor=self.export_scale_factor, transparent=self.export_transparent,
                )

            imgui.end_popup()


    def export_gif(self,
        output_path: str, output_fps: float, scale_factor=None, transparent=True, ensure_no_overwrite=True,
    ):
        assert output_path.split(".")[-1] in ["gif", "webm"], "Output path must be a gif or webm file."

        # self.scene.get_node_by_name('Target mesh').enabled = False
        for i in tqdm(range(self.n_snapshots), desc="Rendering frames"):
            self.scene.get_node_by_name('Snapshots mesh').nodes[i]._enabled = True
            # set the mesh to be visible
            invisible_ids = np.delete(np.arange(self.n_snapshots), i)
            for j in invisible_ids:
                self.scene.get_node_by_name('Snapshots mesh').nodes[j]._enabled = False

            self.render(0, 0, export=True, transparent_background=transparent)
            # img = self.get_snapshot_image(alpha=transparent)
            img = self.get_current_frame_as_image(alpha=transparent)

            # Scale image by the scale factor.
            if scale_factor is not None and scale_factor != 1.0:
                w = int(img.width * scale_factor)
                h = int(img.height * scale_factor)
                img = img.resize((w, h), Image.LANCZOS)
            
            # # 创建一个可以在给定图像上绘制的对象
            # draw = ImageDraw.Draw(img)
            # font = ImageFont.truetype("/media/box/Elements/Exp/umbra/fonts/Roboto/Roboto-Medium.ttf", 30)
            # draw.text(xy=(0.35*img.width, 0.05*img.height), 
            #           text=f"Step {i}, Loss = {self._snapshots[i].error:05f}", 
            #           font=font, fill=(0, 0, 0))
            img_name = os.path.join(os.path.dirname(output_path), f"{i:03d}.png")
            img.save(img_name)

        # create gif.
        png_dir = os.path.dirname(output_path)
        png_files = sorted([os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.endswith('.png')])
        output_gif_name = os.path.join(os.path.dirname(output_path), f"animation_{self.n_snapshots}.gif")
        with imageio.get_writer(output_gif_name, mode='I', fps=output_fps) as writer: 
            for filename in png_files:
                image = imageio.imread(filename)  
                writer.append_data(image)  

        print(f"GIF saved to {os.path.abspath(output_gif_name)}")


    def get_snapshot_image(self, alpha=False):
        if alpha:
            fmt = "RGBA"
            components = 4
        else:
            fmt = "RGB"
            components = 3
        
        fbo = self.renderer.wnd.fbo
        width = self.renderer.wnd.fbo.viewport[2] - self.renderer.wnd.fbo.viewport[0]
        height = self.renderer.wnd.fbo.viewport[3] - self.renderer.wnd.fbo.viewport[1]
        image = Image.frombytes(
            fmt,
            (width, height),
            fbo.read(viewport=self.renderer.wnd.fbo.viewport, alignment=1, components=components),
        )
        if width != self.renderer.wnd.size[0] or height != self.renderer.wnd.size[1]:
            image = image.resize(self.renderer.wnd.size, Image.NEAREST)

        return image.transpose(Image.FLIP_TOP_BOTTOM)
    
    def gui_export_video(self):
        imgui.set_next_window_size(570, 0)

        if imgui.begin_popup_modal(
            "Export Video", flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE 
        )[0]:
            if self.scene.n_frames == 1:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.2)
                self.export_animation = False

            if self.scene.n_frames == 1:
                imgui.pop_style_var()
                self.export_animation = False

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Output settings.
            imgui.text("Output")

            imgui.spacing()
            imgui.text("Format:")
            imgui.same_line()

            if imgui.radio_button("MP4", self.export_format == "mp4"):
                # true if clicked
                self.export_format = "mp4"
            imgui.same_line(spacing=15)
            if imgui.radio_button("AVI", self.export_format == "avi"):
                self.export_format = "avi"

            imgui.spacing()
            if self.export_format == "mp4":
                # imgui.push_style_var(imgui.STYLE_ALPHA, 0.2)
                self.export_transparent = True

            _, self.export_transparent = imgui.checkbox("Transparent background", True)

            # if self.export_format != "gif":
            #     imgui.pop_style_var()
            #     self.export_transparent = False

            # if imgui.is_item_hovered():
            #     imgui.begin_tooltip()
            #     imgui.text("Available only for WEBM format")
            #     imgui.end_tooltip()

            if self.export_format == "mp4":
                max_output_fps = 60.0
            else:
                max_output_fps = 120.0

            self.export_fps = min(self.export_fps, max_output_fps)
            imgui.text(
                f"Duration: [{self.n_snapshots/self.export_fps:.1f}s]"
            )
            fps_changed, self.export_fps = imgui.drag_float(
                label="fps",
                value=self.export_fps,
                # 1.0,
                min_value=2.0,
                max_value=max_output_fps,
                format="%.1f",
            )

            imgui.spacing()
            imgui.spacing()
            imgui.text(
                f"Resolution: [{int(self.window_size[0] * self.export_scale_factor)}x{int(self.window_size[1] * self.export_scale_factor)}]"
            )
            _, self.export_scale_factor = imgui.drag_float(
                label="Scale",
                value=self.export_scale_factor,
                min_value=0.01,
                max_value=1.0,
                change_speed=0.005,
                format="%.2f",
            )

            imgui.same_line(position=440)
            if imgui.button("1x##scale", width=35):
                self.export_scale_factor = 1.0
            imgui.same_line()
            if imgui.button("1/2x##scale", width=35):
                self.export_scale_factor = 0.5
            imgui.same_line()
            if imgui.button("1/4x##scale", width=35):
                self.export_scale_factor = 0.25
            
            imgui.spacing()
            if self.export_format == "mp4":
                imgui.text("Quality: ")
                imgui.same_line()
                if imgui.radio_button("high", self.export_quality == "high"):
                    self.export_quality = "high"
                imgui.same_line()
                if imgui.radio_button("medium", self.export_quality == "medium"):
                    self.export_quality = "medium"
                imgui.same_line()
                if imgui.radio_button("low", self.export_quality == "low"):
                    self.export_quality = "low"


            # Draw a cancel and exit button on the same line using the available space
            button_width = (imgui.get_content_region_available()[0] - imgui.get_style().item_spacing[0]) * 0.5

            # Style the cancel with a grey color
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.5, 0.5, 0.5, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.6, 0.6, 0.6, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.7, 0.7, 0.7, 1.0)

            if imgui.button("Cancel", width=button_width):
                imgui.close_current_popup()

            imgui.pop_style_color()
            imgui.pop_style_color()
            imgui.pop_style_color()

            imgui.same_line()
            if imgui.button("Export", button_width):
                imgui.close_current_popup()
                output_path = os.path.join(
                        self.output_dir,
                        f"animation.{self.export_format}",
                    )
                self.export_video(
                    output_path=output_path, output_fps=self.export_fps,
                    scale_factor=self.export_scale_factor, transparent=self.export_transparent,
                    quality=self.export_quality,
                )

            imgui.end_popup()



    def export_video(self, 
        output_path: str, output_fps: int, scale_factor=None, transparent=True, quality="medium", ensure_no_overwrite=True,
    ):
        assert output_path is not None
        # Load this module to reduce load time
        import skvideo.io

        path_video, path_gif, is_gif = get_video_paths(output_path, ensure_no_overwrite)
        pix_fmt = "yuva420p" if transparent else "yuv420p"
        outputdict = {
            "-pix_fmt": pix_fmt,
            "-vf": "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Avoid error when image res is not divisible by 2.
            "-r": str(output_fps),
        }

        if path_video.endswith("mp4"):
            quality_to_crf = {
                "high": 23,
                "medium": 28,
                "low": 33,
            }
            # MP4 specific options
            outputdict.update(
                {
                    "-c:v": "libx264",
                    "-preset": "slow",
                    "-profile:v": "high",
                    "-level:v": "4.0",
                    "-crf": str(quality_to_crf[quality]),
                }
            )

        writer = skvideo.io.FFmpegWriter(
            path_video,
            inputdict={
                "-framerate": str(output_fps),
            },
            outputdict=outputdict,
        )

        # Store the current camera and create a copy of it if required.
        saved_camera = self.scene.camera

        # Remember viewer data.
        saved_curr_frame = self.scene.current_frame_id
        saved_run_animations = self.run_animations

        animation_range = [0, self.scene.n_frames - 1]
        # Compute duration of the animation at given playback speed
        animation_frames = (animation_range[1] - animation_range[0]) + 1
        # duration = animation_frames / self.playback_fps

        # Setup viewer for rendering the animation
        self.run_animations = True
        self.scene.current_frame_id = animation_range[0]
        self._last_frame_rendered_at = 0

        output_fps = 30
        frames = self.n_snapshots
        dt = 1 / output_fps
        time = 0
        for i in tqdm(range(self.n_snapshots), desc="Rendering frames"):
            self.scene.get_node_by_name('Snapshots mesh').nodes[i]._enabled = True
            # set the mesh to be visible
            invisible_ids = np.delete(np.arange(self.n_snapshots), i)
            for j in invisible_ids:
                self.scene.get_node_by_name('Snapshots mesh').nodes[j]._enabled = False
                
            self.render(0, 0, export=True, transparent_background=transparent)
            img = self.get_current_frame_as_image(alpha=transparent)

            # Scale image by the scale factor.
            if scale_factor is not None and scale_factor != 1.0:
                w = int(img.width * scale_factor)
                h = int(img.height * scale_factor)
                img = img.resize((w, h), Image.LANCZOS)

            # Write the frame to the video writer.
            if output_path is not None:
                writer.writeFrame(np.array(img))

            time += dt

        writer.close()
        print(f"Video saved to {os.path.abspath(output_path)}")



def plotsnapshots(mesh_path, output_dir, screen_size=[960, 1260]):

    meshes = glob.glob(str(Path(mesh_path)/'*.obj'))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    snapshots = []
    for mesh_idx, mesh_path in enumerate(tqdm(meshes)):
        mesh = trimesh.load(mesh_path)
        vertices = np.array(normalize(np.array(mesh.vertices)))
        faces = np.array(mesh.faces)
        snapshots.append(Snapshot(step=mesh_idx, vertices=vertices, faces=faces))

    v = SnapshotViewer(
        size=screen_size, snapshots=snapshots, output_dir=output_dir,
    )
    v.run()


if __name__ == "__main__":

    plotsnapshots(
        mesh_path=Path(__file__).parent.parent/"data/hack_output/seq",
        output_dir=Path(__file__).parent.parent/"data/hack_output/imgs",
    )

