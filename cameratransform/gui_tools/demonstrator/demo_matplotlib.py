import matplotlib.pyplot as plt
import cameratransform as ct
from matplotlib.widgets import Button, Slider, RadioButtons


from demo_includes import Scene9Cubes

def open_demo():
    """
    Open a demo plot with interactive sliders to control camera parameters.
    """
    # Define initial camera objects
    cam1 = ct.Camera(ct.RectilinearProjection(focallength_px=3860, image=[4608, 2592]), ct.SpatialOrientation())
    cam2 = ct.Camera(ct.RectilinearProjection(focallength_px=3860, image=[4608, 2592]), ct.SpatialOrientationYawPitchRoll())
    scene = Scene9Cubes(cam1)
    cam = cam1

    # Create the figure
    fig = plt.figure(0, (10, 5))

    # Update function for slider changes
    def update(val=None):
        """
        Update the camera parameters and plot the scene views.
        """
        # Update camera parameters based on slider values
        for slider, ax in zip(sliders, slider_axs):
            if ax.get_visible():
                setattr(cam, slider.label._text, slider.val)

        # Set the camera center coordinates (needed when the image height/width changes)
        cam.center_x_px = cam.image_width_px/2
        cam.center_y_px = cam.image_height_px/2

        # Clear the axes
        for ax in axes:
            ax.cla()

        # Update the camera in the scene and plot the scene views
        scene.camera = cam
        scene.plotSceneViews(axes)

    # Create the subplot axes for the scene views
    axes = [plt.subplot(241), plt.subplot(242), plt.subplot(245), plt.subplot(246)]

    # Create lists to store the sliders and their axes
    sliders = []
    slider_axs = []
    slider_offset = 0

    # Function to add a slider
    def add_slider(**kwargs):
        """
        Add a slider to control a camera parameter.
        """
        nonlocal slider_offset

        # Create a new slider and add it to the figure
        ax = fig.add_axes([0.7, 0.8-slider_offset*0.035, 0.25, 0.03])
        freq_slider = Slider(ax=ax, **kwargs)
        freq_slider.on_changed(update)

        # Add the slider and its axis to the lists
        sliders.append(freq_slider)
        slider_axs.append(ax)

        # Update the slider offset
        slider_offset += 1

    # Add sliders for image width, image height, and focal length
    add_slider(label='image_width_px', valmin=0, valmax=5000, valinit=4608, valstep=1)
    add_slider(label='image_height_px', valmin=0, valmax=5000, valinit=2592, valstep=1)
    add_slider(label='focallength_px', valmin=0, valmax=cam.focallength_px*2, valinit=cam.focallength_px)
    slider_offset += 3

    # Create radio buttons to switch between spatial orientations
    resetax = fig.add_axes([0.7, 0.8-slider_offset*0.035, 0.25, 0.03*3])
    slider_offset += 1
    radio_buttons = RadioButtons(resetax, ('tilt-heading-roll', 'yaw-pith-roll'))

    # Function to change the spatial orientation
    def change_spatial(label):
        """
        Change the spatial orientation of the camera.
        """
        nonlocal cam
        if label == 'tilt-heading-roll':
            # Show the sliders for heading, tilt, and roll
            slider_axs[3].set_visible(True)
            slider_axs[4].set_visible(True)
            slider_axs[5].set_visible(True)
            # Hide the sliders for yaw, pitch, and roll
            slider_axs[6].set_visible(False)
            slider_axs[7].set_visible(False)
            slider_axs[8].set_visible(False)

            # convert from yaw-pitch-roll to tilt-heading-roll
            sliders[3].set_val(sliders[6].val)
            sliders[4].set_val(sliders[7].val + 90)
            sliders[5].set_val(-sliders[8].val)
            cam = cam1
        elif label == 'yaw-pith-roll':
            # Hide the sliders for heading, tilt, and roll
            slider_axs[3].set_visible(False)
            slider_axs[4].set_visible(False)
            slider_axs[5].set_visible(False)
            # Show the sliders for yaw, pitch, and roll
            slider_axs[6].set_visible(True)
            slider_axs[7].set_visible(True)
            slider_axs[8].set_visible(True)

            # convert from  tilt-heading-roll to yaw-pitch-roll
            sliders[6].set_val(sliders[3].val)
            sliders[7].set_val(sliders[4].val - 90)
            sliders[8].set_val(-sliders[5].val)
            cam = cam2
        update()
    radio_buttons.on_clicked(change_spatial)

    add_slider(label='heading_deg', valmin=-180, valmax=180, valinit=0)
    add_slider(label='tilt_deg', valmin=0, valmax=180, valinit=30)
    add_slider(label='roll_deg', valmin=-180, valmax=180, valinit=0)
    slider_offset -= 3
    add_slider(label='yaw_deg', valmin=-180, valmax=180, valinit=0)
    add_slider(label='pitch_deg', valmin=-90, valmax=90, valinit=-60)
    add_slider(label='roll_deg', valmin=-180, valmax=180, valinit=0)
    slider_offset += 2
    add_slider(label='elevation_m', valmin=0, valmax=20, valinit=10)
    add_slider(label='pos_x_m', valmin=-10, valmax=10, valinit=0)
    add_slider(label='pos_y_m', valmin=-10, valmax=10, valinit=-5.5)
    slider_offset += 1
    resetax = fig.add_axes([0.7, 0.8-slider_offset*0.035, 0.25, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')


    def reset(event):
        for slider in sliders:
            slider.reset()
    button.on_clicked(reset)

    change_spatial('tilt-heading-roll')

    plt.show()


if __name__ == "__main__":
    open_demo()
