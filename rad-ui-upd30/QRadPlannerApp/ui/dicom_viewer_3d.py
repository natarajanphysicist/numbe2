#--- START OF FILE dicom_viewer_3d.py ---

import logging
import numpy as np
from scipy.ndimage import gaussian_filter

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QLabel
from PyQt5.QtCore import Qt

# VTK imports
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkActor,
    vtkImageActor,
)
from vtkmodules.vtkImagingCore import vtkImageShiftScale, vtkImageReslice
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
from vtkmodules.vtkFiltersSources import vtkLineSource, vtkConeSource, vtkCylinderSource, vtkSphereSource
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
# from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper # CRITICALLY ensure this is commented or removed
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper # Ensure vtkSmartVolumeMapper is imported
# from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkGPUVolumeRayCastMapper # CRITICALLY ensure this is commented or removed


from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util import numpy_support
from typing import Optional, Dict, List, Tuple, Any
import sys

logger = logging.getLogger(__name__)

class DicomViewer3DWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0,0,0,0) 

        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.main_layout.addWidget(self.vtkWidget)

        self.ren = vtkRenderer()
        self.ren.SetBackground(0.1, 0.2, 0.4) 
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)

        self.volume_actor: Optional[vtkVolume] = None
        self.tumor_actor: Optional[vtkActor] = None
        self.oar_actors: Dict[str, vtkActor] = {}
        self.dose_isosurface_actors: List[vtkActor] = []
        self.beam_visualization_actors: List[vtkActor] = []
        self.image_properties_for_viz: Optional[Dict] = None
        self.test_slice_actor: Optional[vtkImageActor] = None
        self.test_sphere_actor: Optional[vtkActor] = None

        self.legend_label = QLabel(self)
        self.legend_label.setStyleSheet("background-color: rgba(255,255,255,200); padding: 4px; border: 1px solid #888; font-size: 12px;")
        self.legend_label.setText("<b>Legend:</b><br><span style='color:#ff0000;'>■</span> Tumor<br><span style='color:#00ff00;'>■</span> OAR<br><span style='color:#0000ff;'>■</span> Dose Isosurface")
        self.legend_label.move(10, 10)
        self.legend_label.setFixedWidth(140)
        self.legend_label.setFixedHeight(70)
        self.legend_label.show()

        self.vtkWidget.Initialize()
        logger.info("DicomViewer3DWidget initialized.")

        logger.info("3D View: Setting vtkInteractorStyleTrackballCamera.")
        style = vtkInteractorStyleTrackballCamera()
        interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        if interactor:
            interactor.SetInteractorStyle(style)
        else:
            logger.error("3D View: Failed to get interactor to set style.")


    def _numpy_to_vtkimage(self, np_array_s_r_c: np.ndarray, image_properties: Optional[Dict] = None) -> vtkImageData:
        vtk_image = vtkImageData()
        depth_s, height_r, width_c = np_array_s_r_c.shape
        vtk_image.SetDimensions(width_c, height_r, depth_s)

        if image_properties:
            spacing_x_vtk = image_properties.get('pixel_spacing', [1.0, 1.0])[1]
            spacing_y_vtk = image_properties.get('pixel_spacing', [1.0, 1.0])[0]
            spacing_z_vtk = image_properties.get('slice_thickness', 1.0)
            vtk_image.SetSpacing(spacing_x_vtk, spacing_y_vtk, spacing_z_vtk)
            origin_pat = image_properties.get('origin', [0.0, 0.0, 0.0])
            vtk_image.SetOrigin(origin_pat[0], origin_pat[1], origin_pat[2])
        else: 
            vtk_image.SetSpacing(1.0,1.0,1.0); vtk_image.SetOrigin(0.0,0.0,0.0)

        vtk_array = numpy_support.numpy_to_vtk(num_array=np_array_s_r_c.ravel(order='F'), deep=True)
        vtk_image.GetPointData().SetScalars(vtk_array)
        return vtk_image
    
    def _create_surface_actor_from_mask(self, mask_data_zyx: np.ndarray,
                                        image_properties: Dict, 
                                        color: Tuple[float, float, float],
                                        opacity: float = 0.3) -> Optional[vtkActor]:
        if mask_data_zyx is None or not np.any(mask_data_zyx):
            return None
        try:
            vtk_mask_image = self._numpy_to_vtkimage(mask_data_zyx.astype(np.uint8), image_properties)
            mc = vtkDiscreteMarchingCubes()
            mc.SetInputData(vtk_mask_image)
            mc.SetValue(0, 1)
            mc.Update()
            if mc.GetOutput() is None or mc.GetOutput().GetNumberOfPoints() == 0:
                logger.debug("No surface generated by marching cubes for a mask.")
                return None
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(mc.GetOutputPort())
            mapper.ScalarVisibilityOff()
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color[0], color[1], color[2])
            actor.GetProperty().SetOpacity(opacity)
            return actor
        except Exception as e:
            logger.error(f"Error creating surface actor from mask: {e}", exc_info=True)
            return None

    def update_volume(self, volume_data_full_zyx: Optional[np.ndarray],    # (s,r,c)
                      image_properties: Optional[Dict],
                      tumor_mask_full_zyx: Optional[np.ndarray] = None,   # (s,r,c)
                      oar_masks_full_zyx: Optional[Dict[str, np.ndarray]] = None): # (s,r,c)
        logger.info("3D View: update_volume called. Preparing for Full 3D Volume with SmartMapper[TextureRequested].")
        if hasattr(self, 'legend_label'):
            self.legend_label.show()
        # Cleanup all actors
        if hasattr(self, 'test_sphere_actor') and self.test_sphere_actor:
            self.ren.RemoveActor(self.test_sphere_actor)
            self.test_sphere_actor = None
            logger.info("3D View: Removed test_sphere_actor.")
        if hasattr(self, 'test_slice_actor') and self.test_slice_actor:
            self.ren.RemoveActor(self.test_slice_actor)
            self.test_slice_actor = None
            logger.info("3D View: Removed previous test_slice_actor.")
        if self.volume_actor is not None:
            self.ren.RemoveVolume(self.volume_actor)
            self.volume_actor = None
            logger.info("3D View: Removed main volume_actor.")
        if self.tumor_actor is not None:
            self.ren.RemoveActor(self.tumor_actor)
            self.tumor_actor = None
            logger.info("3D View: Removed tumor_actor.")
        self._clear_oar_actors()
        self._clear_dose_isosurfaces()
        self._clear_beam_visualization()
        logger.info("3D View: All relevant actors cleared for new setup.")

        self.image_properties_for_viz = image_properties

        if volume_data_full_zyx is None or image_properties is None:
            logger.info("No volume data or properties to display in 3D view.")
            if self.vtkWidget.GetRenderWindow():
                self.ren.ResetCamera()
                self.vtkWidget.GetRenderWindow().Render()
            return
        # --- Apply light Gaussian filter to reduce noise ---
        filtered_volume = gaussian_filter(volume_data_full_zyx, sigma=0.7)
        logger.info(f"3D View: Gaussian filter applied (sigma=0.7). Range: {filtered_volume.min():.2f} to {filtered_volume.max():.2f}")
        min_hu_display = -1024.0
        max_hu_display = 3000.0
        volume_data_clipped_zyx = np.clip(filtered_volume, min_hu_display, max_hu_display)
        logger.info(f"3D View: Clipped volume data range for display: {volume_data_clipped_zyx.min():.2f} to {volume_data_clipped_zyx.max():.2f}")
        vtk_volume_image = self._numpy_to_vtkimage(volume_data_clipped_zyx.astype(np.float32), image_properties)
        try:
            scalar_range = vtk_volume_image.GetScalarRange()
            logger.info(f"3D View: VTK Volume Image Scalar Range: {scalar_range[0]:.2f} to {scalar_range[1]:.2f}")
        except Exception as e_sr:
            logger.warning(f"3D View: Could not get scalar range from vtk_volume_image: {e_sr}")

        # --- START OF FULL 3D VOLUME RENDERING ---
        logger.info("3D View: Attempting to display full 3D CT volume.")
        if vtk_volume_image is not None and vtk_volume_image.GetNumberOfPoints() > 0:
            # --- Refined color and opacity transfer functions for smoother look ---
            color_func = vtkColorTransferFunction()
            opacity_func = vtkPiecewiseFunction()
            color_func.RemoveAllPoints()
            color_func.AddRGBPoint(min_hu_display, 0.0, 0.0, 0.0)
            color_func.AddRGBPoint(-500.0, 0.2, 0.2, 0.2)
            color_func.AddRGBPoint(0.0, 0.5, 0.5, 0.5)
            color_func.AddRGBPoint(300.0, 0.7, 0.7, 0.7)
            color_func.AddRGBPoint(1000.0, 0.9, 0.9, 0.9)
            color_func.AddRGBPoint(max_hu_display, 1.0, 1.0, 1.0)
            opacity_func.RemoveAllPoints()
            opacity_func.AddPoint(min_hu_display, 0.0)
            opacity_func.AddPoint(-500.0, 0.01)
            opacity_func.AddPoint(0.0, 0.07)
            opacity_func.AddPoint(300.0, 0.15)
            opacity_func.AddPoint(1000.0, 0.4)
            opacity_func.AddPoint(max_hu_display, 0.6)

            if not hasattr(self, 'volume_property') or self.volume_property is None:
                 self.volume_property = vtkVolumeProperty()
            self.volume_property.SetColor(color_func)
            self.volume_property.SetScalarOpacity(opacity_func)
            self.volume_property.SetInterpolationTypeToLinear()
            self.volume_property.ShadeOn()
            self.volume_property.SetAmbient(0.3)
            self.volume_property.SetDiffuse(0.7)
            self.volume_property.SetSpecular(0.2)
            self.volume_property.SetSpecularPower(10.0)

            # --- Use GPU Ray Casting if available ---
            try:
                from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkGPUVolumeRayCastMapper
                volume_mapper = vtkGPUVolumeRayCastMapper()
                volume_mapper.SetInputData(vtk_volume_image)
                volume_mapper.SetBlendModeToComposite()
                volume_mapper.SetSampleDistance(0.2)
            except ImportError:
                from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
                volume_mapper = vtkSmartVolumeMapper()
                volume_mapper.SetInputData(vtk_volume_image)
                if hasattr(volume_mapper, 'SetBlendModeToComposite'):
                    volume_mapper.SetBlendModeToComposite()
                if hasattr(volume_mapper, 'SetSampleDistance'):
                    volume_mapper.SetSampleDistance(0.5)

            self.volume_actor = vtkVolume()
            self.volume_actor.SetMapper(volume_mapper)
            self.volume_actor.SetProperty(self.volume_property)

            self.ren.AddVolume(self.volume_actor)
            logger.info("3D View: Main 3D vtkVolume actor configured and added to renderer.")
            try:
                bounds = self.volume_actor.GetBounds()
                logger.info(f"3D View: Full 3D Volume actor bounds: {bounds}")
            except Exception as e_bounds:
                logger.warning(f"3D View: Could not get full 3D volume actor bounds: {e_bounds}")
        else:
            logger.warning("3D View: Main 3D vtk_volume_image not available or empty, cannot display full 3D volume.")
        # --- END OF FULL 3D VOLUME RENDERING ---

        if tumor_mask_full_zyx is not None and np.any(tumor_mask_full_zyx):
            # Increase opacity for better visibility
            self.tumor_actor = self._create_surface_actor_from_mask(
                tumor_mask_full_zyx, image_properties, color=(1.0, 0.0, 0.0), opacity=0.7
            )
            if self.tumor_actor:
                self.ren.AddActor(self.tumor_actor)
                logger.info("Tumor mask actor added (opacity 0.7 for visibility).")

        if oar_masks_full_zyx:
            oar_colors = [(0,1,0), (0,0,1), (1,1,0), (0,1,1), (1,0,1), (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5)]
            color_idx = 0
            for name, oar_mask_zyx_data in oar_masks_full_zyx.items():
                if oar_mask_zyx_data is None or not np.any(oar_mask_zyx_data): continue
                oar_actor = self._create_surface_actor_from_mask(
                    oar_mask_zyx_data, image_properties,
                    color=oar_colors[color_idx % len(oar_colors)], opacity=0.25
                )
                if oar_actor:
                    self.ren.AddActor(oar_actor); self.oar_actors[name] = oar_actor
                    logger.info(f"OAR actor for '{name}' added."); color_idx += 1

        if self.vtkWidget.GetRenderWindow():
            self.ren.ResetCamera()
            self.ren.ResetCameraClippingRange()
            try:
                camera = self.ren.GetActiveCamera()
                if camera:
                    logger.info(f"3D View: Camera Position: {camera.GetPosition()}, FocalPoint: {camera.GetFocalPoint()}, ViewUp: {camera.GetViewUp()}, ClipRange: {camera.GetClippingRange()}, ViewAngle: {camera.GetViewAngle()}")
                else:
                    logger.warning("3D View: No active camera found to log parameters.")
            except Exception as e_cam:
                logger.warning(f"3D View: Could not get camera parameters: {e_cam}")
            self.vtkWidget.GetRenderWindow().Render()
        logger.info("3D View updated (Full 3D Volume with SmartMapper [Default Mode] and 'more solid' TFs).")

    def _clear_oar_actors(self):
        logger.debug(f"Clearing {len(self.oar_actors)} OAR actors.")
        for actor in self.oar_actors.values():
            self.ren.RemoveActor(actor)
        self.oar_actors.clear()

    def _clear_dose_isosurfaces(self):
        logger.debug(f"Clearing {len(self.dose_isosurface_actors)} dose isosurface actors.")
        for actor in self.dose_isosurface_actors:
            self.ren.RemoveActor(actor)
        self.dose_isosurface_actors.clear()

    def _update_dose_isosurfaces(self, dose_volume_full_crs: Optional[np.ndarray],
                                 image_properties: Optional[Dict],
                                 isovalues_list: Optional[List[float]] = None):
        logger.info("Updating dose isosurfaces...")
        self._clear_dose_isosurfaces()

        if dose_volume_full_crs is None or image_properties is None or not isovalues_list:
            logger.info("No dose volume, properties, or isovalues provided. Isosurfaces cleared or not generated.")
            if self.vtkWidget.GetRenderWindow(): self.vtkWidget.GetRenderWindow().Render()
            return

        try:
            logger.debug(f"Dose volume for isosurfaces shape (c,r,s): {dose_volume_full_crs.shape}, dtype: {dose_volume_full_crs.dtype}")
            dose_volume_full_zyx = np.transpose(dose_volume_full_crs, (2, 1, 0)).astype(np.float32)
            dose_vtk_image = self._numpy_to_vtkimage(dose_volume_full_zyx, image_properties)

            colors_vtk_dose = [
                (1,0,0, 0.3), (0,1,0, 0.3), (0,0,1, 0.3), (1,1,0, 0.3),
                (0,1,1, 0.3), (1,0,1, 0.3), (1,0.5,0, 0.3), (0.5,1,0, 0.3),
                (0,0.5,1, 0.3), (0.5,0,1, 0.3) 
            ]

            for i, value in enumerate(isovalues_list):
                if not isinstance(value, (int, float)): logger.warning(f"Skipping invalid isovalue: {value}"); continue
                logger.debug(f"Creating isosurface for value: {value} Gy")
                contour_filter = vtkMarchingCubes()
                contour_filter.SetInputData(dose_vtk_image)
                contour_filter.SetValue(0, value)
                contour_filter.Update()
                if contour_filter.GetOutput() is None or contour_filter.GetOutput().GetNumberOfPoints() == 0:
                    logger.info(f"No geometry for isovalue {value} Gy. Skipping."); continue
                mapper = vtkPolyDataMapper(); mapper.SetInputConnection(contour_filter.GetOutputPort()); mapper.ScalarVisibilityOff()
                actor = vtkActor(); actor.SetMapper(mapper)
                color_def = colors_vtk_dose[i % len(colors_vtk_dose)]
                actor.GetProperty().SetColor(color_def[0], color_def[1], color_def[2])
                actor.GetProperty().SetOpacity(color_def[3]) 
                self.ren.AddActor(actor); self.dose_isosurface_actors.append(actor)
                logger.info(f"Added isosurface actor for {value} Gy.")
        except Exception as e:
            logger.error(f"Error creating dose isosurfaces: {e}", exc_info=True)

        if self.vtkWidget.GetRenderWindow():
            self.ren.ResetCameraClippingRange()
            self.vtkWidget.GetRenderWindow().Render()
        logger.info("Dose isosurfaces update process finished.")        

    def _planner_coords_to_patient_world_coords(self, planner_coords_crs: np.ndarray) -> Optional[np.ndarray]:
        if self.image_properties_for_viz is None:
            logger.error("Cannot convert planner to world coords: image_properties_for_viz not set.")
            return None
        
        col_idx, row_idx, slice_idx = planner_coords_crs

        col_spacing = self.image_properties_for_viz.get('pixel_spacing', [1.0,1.0])[1]
        row_spacing = self.image_properties_for_viz.get('pixel_spacing', [1.0,1.0])[0]
        slice_spacing = self.image_properties_for_viz.get('slice_thickness', 1.0)
        
        img_x_disp = col_idx * col_spacing
        img_y_disp = row_idx * row_spacing
        img_z_disp = slice_idx * slice_spacing
        
        scaled_img_coords_vec = np.array([img_x_disp, img_y_disp, img_z_disp], dtype=float)
        
        orientation_matrix = np.array(self.image_properties_for_viz.get('orientation_matrix_3x3', np.eye(3)))
        origin_patient_lps = np.array(self.image_properties_for_viz.get('origin', [0.0,0.0,0.0]))
        
        world_coords_lps = origin_patient_lps + orientation_matrix @ scaled_img_coords_vec

        logger.debug(f"Planner coords (c,r,s): {planner_coords_crs} -> World coords (LPS): {world_coords_lps}")
        return world_coords_lps

    def display_beams(self, beam_viz_data: Optional[Dict[str, Any]]):
        self._clear_beam_visualization()
        if beam_viz_data is None or self.image_properties_for_viz is None:
            logger.info("No beam viz data or image_properties. Beams not displayed.")
            if self.vtkWidget.GetRenderWindow(): self.vtkWidget.GetRenderWindow().Render()
            return

        beam_directions_planner = beam_viz_data.get("beam_directions", [])
        beam_weights = beam_viz_data.get("beam_weights", np.array([]))
        source_positions_planner_vox = beam_viz_data.get("source_positions_planner_coords", [])
        isocenter_planner_vox = np.array(beam_viz_data.get("isocenter_planner_coords", [0.0,0.0,0.0]))

        isocenter_world_lps = self._planner_coords_to_patient_world_coords(isocenter_planner_vox)
        if isocenter_world_lps is None:
            logger.error("Failed to convert isocenter to world coordinates for beam display."); return

        logger.info(f"Displaying beams. Isocenter (world LPS): {isocenter_world_lps}. Num beams: {len(beam_directions_planner)}")
        max_weight = np.max(beam_weights) if beam_weights.size > 0 and np.max(beam_weights) > 1e-6 else 1.0

        for i, direction_vec_planner in enumerate(beam_directions_planner):
            weight = beam_weights[i] if i < len(beam_weights) else 0
            if weight < 0.01: continue

            source_pos_planner_vox_np = np.array(source_positions_planner_vox[i])
            source_pos_world_lps = self._planner_coords_to_patient_world_coords(source_pos_planner_vox_np)
            if source_pos_world_lps is None:
                logger.warning(f"Could not convert source for beam {i} to world. Skipping."); continue

            line_source = vtkLineSource()
            line_source.SetPoint1(source_pos_world_lps[0], source_pos_world_lps[1], source_pos_world_lps[2])
            line_source.SetPoint2(isocenter_world_lps[0], isocenter_world_lps[1], isocenter_world_lps[2])
            line_source.Update()

            mapper = vtkPolyDataMapper(); mapper.SetInputConnection(line_source.GetOutputPort())
            actor = vtkActor(); actor.SetMapper(mapper)
            
            intensity = weight / max_weight
            actor.GetProperty().SetColor(intensity, 1.0 - intensity, 0.0)
            actor.GetProperty().SetLineWidth(1.0 + intensity * 3.0)
            actor.GetProperty().SetOpacity(0.6 + intensity * 0.4)

            self.ren.AddActor(actor); self.beam_visualization_actors.append(actor)
            logger.debug(f"Beam {i}: src_world={source_pos_world_lps}, iso_world={isocenter_world_lps}, weight={weight:.2f}")

        if self.vtkWidget.GetRenderWindow(): self.vtkWidget.GetRenderWindow().Render()
        logger.info("Beam visualization updated.")

    def _clear_beam_visualization(self):
        logger.debug(f"Clearing {len(self.beam_visualization_actors)} beam visualization actors.")
        for actor in self.beam_visualization_actors:
            self.ren.RemoveActor(actor)
        self.beam_visualization_actors.clear()
    
    def clear_view(self):
        logger.info("Clearing 3D viewer (volume, tumor, OARs, beams, and dose isosurfaces).")
        if self.volume_actor is not None: self.ren.RemoveVolume(self.volume_actor); self.volume_actor = None
        if self.test_slice_actor is not None: self.ren.RemoveActor(self.test_slice_actor); self.test_slice_actor = None
        if self.test_sphere_actor is not None: self.ren.RemoveActor(self.test_sphere_actor); self.test_sphere_actor = None
        if self.tumor_actor is not None: self.ren.RemoveActor(self.tumor_actor); self.tumor_actor = None
        self._clear_oar_actors()
        self._clear_dose_isosurfaces()
        self._clear_beam_visualization() 
        if self.vtkWidget.GetRenderWindow(): self.vtkWidget.GetRenderWindow().Render()
        if hasattr(self, 'legend_label'):
            self.legend_label.hide()

    def set_window_level(self, wc, ww):
        min_hu = wc - ww / 2.0
        max_hu = wc + ww / 2.0
        color_func = vtkColorTransferFunction()
        opacity_func = vtkPiecewiseFunction()
        color_func.RemoveAllPoints()
        color_func.AddRGBPoint(min_hu, 0.0, 0.0, 0.0)
        color_func.AddRGBPoint(wc, 0.8, 0.8, 0.8)
        color_func.AddRGBPoint(max_hu, 1.0, 1.0, 1.0)
        opacity_func.RemoveAllPoints()
        opacity_func.AddPoint(min_hu, 0.0)
        opacity_func.AddPoint(wc, 0.18)
        opacity_func.AddPoint(max_hu, 0.7)
        if not hasattr(self, 'volume_property') or self.volume_property is None:
            return
        self.volume_property.SetColor(color_func)
        self.volume_property.SetScalarOpacity(opacity_func)
        self.volume_property.SetInterpolationTypeToLinear()
        self.volume_property.ShadeOn()
        self.volume_property.SetAmbient(0.3)
        self.volume_property.SetDiffuse(0.7)
        self.volume_property.SetSpecular(0.2)
        self.volume_property.SetSpecularPower(10.0)
        if self.vtkWidget.GetRenderWindow():
            self.vtkWidget.GetRenderWindow().Render()

    def set_preset(self, preset):
        if preset == 'bone':
            self.set_window_level(300, 1500)
        elif preset == 'soft':
            self.set_window_level(40, 400)
        elif preset == 'brain':
            self.set_window_level(40, 80)

    def reset_view(self):
        if self.ren:
            self.ren.ResetCamera()
            self.ren.ResetCameraClippingRange()
            if self.vtkWidget.GetRenderWindow():
                self.vtkWidget.GetRenderWindow().Render()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    viewer3d = DicomViewer3DWidget()
    vol_shape_zyx = (50, 64, 64)
    dummy_volume_data_zyx = np.random.randint(-1000, 2000, size=vol_shape_zyx, dtype=np.int16)
    dummy_tumor_mask_zyx = np.zeros(vol_shape_zyx, dtype=bool); dummy_tumor_mask_zyx[20:30, 25:35, 25:35] = True
    dummy_oar_masks_zyx = {"OAR1": np.zeros(vol_shape_zyx, dtype=bool)}; dummy_oar_masks_zyx["OAR1"][15:25, 10:20, 40:50] = True
    
    dummy_image_properties = {
        'pixel_spacing': [0.8, 0.9],
        'slice_thickness': 2.5,
        'origin': [-100.0, -120.0, -80.0],
        'orientation_matrix_3x3': np.eye(3).tolist()
    }
    
    viewer3d.update_volume(dummy_volume_data_zyx,
                           dummy_image_properties,
                           tumor_mask_full_zyx=dummy_tumor_mask_zyx, 
                           oar_masks_full_zyx=dummy_oar_masks_zyx)
    
    dose_shape_crs = (vol_shape_zyx[2], vol_shape_zyx[1], vol_shape_zyx[0])
    dummy_dose_crs = np.zeros(dose_shape_crs, dtype=np.float32)
    cx,cy,cz = [d//2 for d in dose_shape_crs]; radius_dose = min(cx,cy,cz)//2
    x,y,z = np.ogrid[:dose_shape_crs[0], :dose_shape_crs[1], :dose_shape_crs[2]]
    mask_sphere_crs = (x-cx)**2 + (y-cy)**2 + (z-cz)**2 <= radius_dose**2
    dummy_dose_crs[mask_sphere_crs] = 60.0; dummy_dose_crs += np.random.rand(*dose_shape_crs)*10
    viewer3d._update_dose_isosurfaces(dummy_dose_crs, dummy_image_properties, isovalues_list=[20.0, 40.0, 55.0])

    planner_grid_size_crs = dose_shape_crs 
    isocenter_planner_vox_test = np.array([planner_grid_size_crs[0]//2, planner_grid_size_crs[1]//2, planner_grid_size_crs[2]//2])
    
    beam_viz_test_data = {
        "beam_directions": [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0), (0.707, 0.707, 0)],
        "beam_weights": np.array([1.0, 0.8, 0.6, 0.4, 0.9]),
        "source_positions_planner_coords": [
            (isocenter_planner_vox_test - np.array([1,0,0])*100).tolist(),
            (isocenter_planner_vox_test - np.array([0,1,0])*100).tolist(),
            (isocenter_planner_vox_test - np.array([-1,0,0])*100).tolist(),
            (isocenter_planner_vox_test - np.array([0,-1,0])*100).tolist(),
            (isocenter_planner_vox_test - np.array([0.707,0.707,0])*100).tolist()
        ],
        "isocenter_planner_coords": isocenter_planner_vox_test.tolist()
    }
    viewer3d.display_beams(beam_viz_test_data)

    viewer3d.resize(800, 600)
    viewer3d.show()
    sys.exit(app.exec_())
