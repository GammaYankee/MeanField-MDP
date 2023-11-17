import matplotlib.cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import time
import imageio
import moviepy.video.io.ImageSequenceClip
import os
from mf_mdp.envs.traffic import TrafficEnv


class Renderer():
    def __init__(self):
        pass

    def initialize_from_config(self, config_data, section_name):
        pass

    def create_figure(self):
        pass

    def render_all(self):
        pass

    def render_agents(self):
        pass

    def render_obstacles(self):
        pass

    def render_trajectories(self):
        pass


class MatplotlibRenderer(Renderer):
    def __init__(self, xaxis_range=None, yaxis_range=None, auto_range=None, figure_size=None, figure_dpi=None,
                 axis_equal=True, show_axis=True,
                 save_gif=False, save_dir=None, hold_time=0.3):
        Renderer.__init__(self)
        self.xaxis_range = xaxis_range
        self.yaxis_range = yaxis_range
        self.auto_range = auto_range
        self.figure_size = figure_size
        self.figure_dpi = figure_dpi
        self.axis_equal = axis_equal
        self.show_axis = show_axis
        self._figure = None
        self._axis = None
        self._hold_time = hold_time

        # for saving animation
        self.save_gif = save_gif
        self.save_dir = save_dir
        self.frame = int(0)
        self.episode = int(0)

        self.show_cbar = False
        self.cmap = None
        self.norm = None

    def create_figure(self):
        self._figure = plt.figure(figsize=(self.figure_size[0], self.figure_size[1]), dpi=self.figure_dpi)
        self._axis = self._figure.add_subplot(1, 1, 1)
        self._axis.grid(False)
        if self.axis_equal:
            self._axis.set_aspect('equal', adjustable='box')
        if not self.show_axis:
            self._axis.axis('off')
        plt.figure(self._figure.number)

        if self.show_cbar:
            sm = plt.cm.ScalarMappable(cmap=self.cmap)
            sm.set_array([])
            self._figure.colorbar(sm, ticks=np.linspace(0, 1, 6), norm=self.norm,
                                  pad=0.2, shrink=0.5)

    def set_range(self):
        if not self.auto_range:
            self._axis.axis([self.xaxis_range[0], self.xaxis_range[1], self.yaxis_range[0], self.yaxis_range[1]])
        plt.grid(False)

    def show(self):
        if not self.show_axis:
            self._axis.axis('off')
        self.set_range()
        plt.pause(0.01)
        if self.save_gif:
            self.save()
        self.frame += 1

    def clear(self):
        plt.cla()

    def reset(self):
        self.episode += 1
        self.frame = 0

    def save(self, save_path_name=None):
        if self.save_dir is not None and not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        self.set_range()
        if save_path_name is None:
            assert self.save_dir is not None
            save_path_name = self.save_dir / 'ep{}-frame-{}.png'.format(self.episode, self.frame)
        plt.savefig(save_path_name)

    def hold(self, t=None):
        if t is not None:
            time.sleep(t)
        else:
            time.sleep(self._hold_time)

    def render_gif(self, episode=None, duration=0.3):
        if episode is None:
            episode = 0

        # get n_frames
        n_frames = 0
        for x in os.listdir(self.save_dir):
            if x.startswith("ep{}".format(episode)) and x.endswith(".png"):
                file_name = x.split(".")[0]
                frame = int(file_name.split("-")[-1])
                if frame > n_frames:
                    n_frames = frame
        frames = [0 for _ in range(3)]
        for frame in range(n_frames + 1):
            frames.append(frame)
        frames += [n_frames for _ in range(3)]

        images = []
        for frame in frames:
            file_name = self.save_dir / 'ep{}-frame-{}.png'.format(episode, frame)
            images.append(imageio.imread(file_name))

        gif_dir = self.save_dir / 'ep{}-movie.gif'.format(episode)
        imageio.mimsave(gif_dir, images, duration=duration)

    def render_mp4(self, episode=None, duration=0.3):
        if episode is None:
            episode = 0

        # get n_frames
        n_frames = 0
        for x in os.listdir(self.save_dir):
            if x.startswith("ep{}".format(episode)) and x.endswith(".png"):
                file_name = x.split(".")[0]
                frame = int(file_name.split("-")[-1])
                if frame > n_frames:
                    n_frames = frame

        image_folder = self.save_dir
        fps = int(1 / duration)

        frames = [0 for _ in range(3)]
        for frame in range(n_frames + 1):
            frames.append(frame)
        frames += [n_frames for _ in range(3)]

        image_files = [os.path.join(image_folder, 'ep{}-frame-{}.png'.format(episode, frame))
                       for frame in frames]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(str(self.save_dir / 'ep{}-movie.mp4'.format(episode)))


class TrafficRenderer(MatplotlibRenderer):
    def __init__(self, env: TrafficEnv, lane_width=0.5,
                 show_cbar=True, cmap_type="jet",
                 save_gif=False, save_dir=None, show_axis=False):
        self.env = env
        self.n_lanes, self.n_blocks = env.n_lanes, env.n_blocks
        self.lane_width = lane_width

        super().__init__(xaxis_range=[0, lane_width * self.n_lanes], yaxis_range=[0, self.n_blocks], auto_range=False,
                         figure_size=[4, 4], figure_dpi=240, show_axis=show_axis,
                         save_gif=save_gif, save_dir=save_dir)

        # for color bar
        self.show_cbar = show_cbar
        self.norm = matplotlib.colors.Normalize(vmin=0, vmax=1 / self.env.n_lanes)
        self.cmap = matplotlib.cm.get_cmap(cmap_type)

    def render_lanes(self):
        # add boundary
        self._axis.add_artist(lines.Line2D([0, 0], [0, self.n_blocks], color="k", linestyle="-", linewidth=1))
        self._axis.add_artist(lines.Line2D([self.lane_width * self.n_lanes, self.lane_width * self.n_lanes],
                                           [0, self.n_blocks], color="k", linestyle="-", linewidth=1))
        self._axis.add_artist(lines.Line2D([0, self.lane_width * self.n_lanes], [0, 0],
                                           color="k", linestyle="-", linewidth=1))
        self._axis.add_artist(lines.Line2D([0, self.lane_width * self.n_lanes],
                                           [self.n_blocks, self.n_blocks], color="k", linestyle="-", linewidth=1))

        # mark lanes
        for i in range(1, self.n_lanes):
            self._axis.add_artist(lines.Line2D([i * self.lane_width, self.lane_width * i],
                                               [0, self.n_blocks], color="k", linestyle=":", linewidth=0.7))

        # mark blocks
        for j in range(1, self.n_blocks):
            self._axis.add_artist(lines.Line2D([0, self.lane_width * self.n_lanes],
                                               [j, j], color="k", linestyle=":", linewidth=0.3))

    def render_obstacles(self):
        for s in self.env.obstacle_state_list:
            lane, block = self.env.state2status(s)
            self._axis.add_patch(patches.Rectangle((lane * self.lane_width, block), self.lane_width, 1.0,
                                                   facecolor='k'))

    def render_car(self, s, color):
        lane, block = self.env.state2status(s)
        self._axis.add_patch(patches.Rectangle(((lane + 0.2) * self.lane_width, block + 0.2), self.lane_width / 3, 0.3,
                                               facecolor=color))

    def render_agents(self, agent_list):
        self.render_lanes()
        for agent in agent_list:
            self.render_car(s=agent.state, color=agent.color)

    def render_mf(self, mf):

        for s in range(self.env.n_states):
            p = mf[s] * self.env.n_lanes
            lane, block = self.env.state2status(s)
            rgba = self.cmap(p)
            self._axis.add_patch(
                patches.Rectangle((lane * self.lane_width, block), self.lane_width, 1.0, facecolor=rgba))


if __name__ == "__main__":
    action_vector_list = [np.array([-1, 0]), np.array([-1, 1]), np.array([0, 1]), np.array([1, 1]), np.array([1, 0])]
    mf_env = TrafficEnv(n_lanes=3, n_blocks=5, cells_per_block=1, n_actions=5, action_vec_list=action_vector_list, Tf=3)
    renderer = TrafficRenderer(env=mf_env)

    renderer.create_figure()
    renderer.render_lanes()
    renderer.show()
    renderer.hold(t=10)
    renderer.clear()
