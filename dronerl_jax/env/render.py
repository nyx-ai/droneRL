from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Dict, Tuple, Literal
import subprocess
import itertools
import copy
import numpy as np
import os
from collections import deque

from .constants import Object, Action


assets_path = os.path.dirname(os.path.realpath(__file__))
font_path = os.path.join(assets_path, "assets", "font", "Press_Start_2P", "PressStart2P-Regular.ttf")
sprite_path = os.path.join(assets_path, "assets", "16ShipCollection.png")


class Renderer:
    def __init__(
            self,
            n_drones: int,
            grid_size: int,
            player_name_mappings: Optional[Dict[int, str]] = None,
            rgb_render_rescale: float = 1.0,
            trace_length: int = 0,
            trace_drone_ids_only: Tuple[int] = (0,),
            image_format: Literal['png', 'jpg'] = 'png'):
        self.n_drones = n_drones
        self.grid_size = grid_size
        self.player_name_mappings = player_name_mappings
        self.rgb_render_rescale = rgb_render_rescale
        self.orientation = self.n_drones * [Action.RIGHT]
        self.image_format = image_format
        self.is_initialized = False
        self.font = ImageFont.truetype(font_path, 8)
        self.line_spacing = 16
        self.large_line_spacing = 22
        self.traces = [deque(maxlen=trace_length) for _ in range(n_drones)]
        self.trace_length = trace_length
        self.trace_paths = [[None for _ in range(grid_size)] for _ in range(grid_size)]
        self.trace_drone_ids_only = trace_drone_ids_only
        self.ship_colors = []

    def init(self):
        # Load RGB image
        sprites_img = Image.open(sprite_path)
        sprites_img_array = np.array(sprites_img)

        # Make black background transparent
        black_pixels = (sprites_img_array[:, :, 0] + sprites_img_array[:, :, 1] + sprites_img_array[:, :, 2]) == 0
        sprites_img_array[np.nonzero(black_pixels) + (3,)] = 0

        # Create tiles with the standard objects
        def get_ships_tile(row, col):
            tiles_size, small_padding, big_padding = 16, 4, 10
            top_corner = (42, 28)

            i = top_corner[0] + row * (tiles_size + small_padding)
            j = top_corner[1] + (col % 5) * (tiles_size + small_padding) + (col // 5) * (
                    5 * (tiles_size + small_padding) + big_padding)
            return Image.fromarray(sprites_img_array[i:i + tiles_size, j:j + tiles_size])

        self.tiles = {
            'packet': get_ships_tile(11, 9),
            'dropzone': get_ships_tile(11, 8),
            'station': get_ships_tile(18, 15),
            'skyscraper': get_ships_tile(18, 12)
        }
        self.traces = [deque(maxlen=self.trace_length) for _ in range(self.n_drones)]
        self.trace_paths = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Define list of ships and colors
        ship_types = [(1, 2), (6, 3), (8, 0), (9, 3), (9, 4), (10, 2), (17, 3)]
        ship_colors = [0, 5, 10, 15, 20]

        # Shuffle them
        shuffled_ships_by_color = []
        for color_col in ship_colors:
            # Make sure we alternate between ships
            idx = np.arange(len(ship_types))
            np.random.shuffle(idx)
            shuffled = np.array(ship_types)[idx]
            shuffled[:, 1] += color_col
            shuffled_ships_by_color.append(shuffled.tolist())

        shuffled_ships = []
        for ships in zip(*shuffled_ships_by_color):
            shuffled_ships.extend(ships)

        # Create iterator
        ships_iter = itertools.cycle(iter(shuffled_ships))

        # Create drone tiles
        def overlay(img_a, img_b, orientation, offset = 7):
            overlay = Image.new('RGB', [img_a.size[0] + offset, img_a.size[1] + offset])
            if orientation == Action.RIGHT:
                overlay.paste(img_a, (offset, 0), img_a)  # move package to the right
                overlay.paste(img_b, (0, 0), img_b)
            elif orientation == Action.DOWN:
                overlay.paste(img_a, (0, offset), img_a)  # move package down
                overlay.paste(img_b, (0, 0), img_b)
            elif orientation == Action.UP:
                overlay.paste(img_a, (0, 0), img_a)
                overlay.paste(img_b, (0, offset), img_b)  # move drone down
            else:
                # left
                overlay.paste(img_a, (0, 0), img_a)
                overlay.paste(img_b, (offset, 0), img_b)  # move drone to right
            return overlay

        self.ship_colors = []
        for drone_idx in range(self.n_drones):
            i, j = next(ships_iter)
            self.ship_colors.append(['blue', 'green', 'red', 'yellow', 'purple'][drone_idx % 5])
            tile = get_ships_tile(i, j)
            for rot, orientation in zip([0, 90, 180, 270], [Action.RIGHT, Action.UP, Action.LEFT, Action.DOWN]):
                label = f'drone_{drone_idx}_{orientation}'
                tile_rot = copy.copy(tile).rotate(rot)  # counter-clock wise
                self.tiles[label] = tile_rot
                self.tiles[label + '_packet'] = overlay(self.tiles['packet'], tile_rot, orientation)
                self.tiles[label + '_charging'] = overlay(self.tiles['station'], tile_rot, orientation, offset=0)
                self.tiles[label + '_dropzone'] = overlay(self.tiles['dropzone'], tile_rot, orientation, offset=0)

        # Create empty frame
        self.render_padding, self.tiles_size = 10, 16
        frames_size = self.tiles_size * self.grid_size + self.render_padding * (self.grid_size + 1)
        self.empty_frame = np.full(shape=(frames_size, frames_size, 3), fill_value=0, dtype=np.uint8)

        # Side panel
        background_color = 'lightblue'
        panel_width = 120
        max_drones = 6
        display_num_drones = min(self.n_drones, max_drones)
        panel_height = display_num_drones * self.large_line_spacing + self.render_padding
        self.panel = Image.new('RGB', (panel_width, panel_height), color=background_color)
        draw_handle = ImageDraw.Draw(self.panel, mode='RGB')

        for drone_idx in range(display_num_drones):
            drone_sprite = self.tiles['drone_{}_{}'.format(drone_idx, Action.RIGHT)]
            sprite_x = self.render_padding
            sprite_y = drone_idx * self.large_line_spacing + self.render_padding
            self.panel.paste(drone_sprite, (sprite_x, sprite_y), drone_sprite)

            player_name = f'Player {drone_idx}'
            if self.player_name_mappings is not None:
                # Optional setting for rendering videos on AIcrowd evaluator
                player_name = self.player_name_mappings.get(drone_idx, '')

            # Print text
            text_x = sprite_x + self.tiles_size + self.render_padding
            text_y = sprite_y + 4
            draw_handle.text((text_x, text_y), player_name, fill='black', font=self.font)

        # Legend
        legend_width = 20
        self.legend = Image.new('RGB', (self.empty_frame.shape[0] + panel_width, legend_width), color=background_color)
        draw_handle = ImageDraw.Draw(self.legend, mode='RGB')
        current_space = 2
        for tile_name, legend_str in zip(['skyscraper', 'station', 'dropzone', 'packet'], ['Skyscraper', 'C. station', 'Dropzone', 'Package']):
            sprite = self.tiles[tile_name]
            self.legend.paste(sprite, (current_space, 2), sprite)
            current_space += 20
            draw_handle.text((current_space, 7), legend_str, fill='black', font=self.font)
            current_space += len(legend_str) * 8 + 7

        # Metrics
        self.metric_panel = Image.new('RGB', (panel_width, self.empty_frame.shape[1] - panel_height), color=background_color)

        # mark as initialized
        self.is_initialized = True


    def render_frame(
            self,
            step: int,
            ground: np.ndarray,
            air: np.ndarray,
            carrying_package: np.ndarray,
            charge: np.ndarray,
            rewards: np.ndarray,
            actions: np.ndarray):
        if not self.is_initialized:
            raise Exception('Renderer was not yet initialized. Before running this method, first run `renderer.init()`.')
        frame = Image.fromarray(self.empty_frame.copy())
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Check tile
                ground_pos = ground[i, j]
                air_pos = air[i, j]
                has_ground_pos = (ground_pos != 0) and (ground_pos is not None)
                has_air_pos = air_pos is not None
                charge_bar = None
                if has_air_pos:
                    drone_orientation = self.orientation[air_pos]
                    if actions[air_pos] != Action.STAY:
                        drone_orientation = actions[air_pos]
                        self.orientation[air_pos] = drone_orientation
                    if carrying_package[air_pos]:
                        tile = self.tiles[f'drone_{air_pos}_{drone_orientation}_packet']
                    else:
                        if ground_pos == Object.STATION:
                            tile = self.tiles[f'drone_{air_pos}_{drone_orientation}_charging']
                        elif ground_pos == Object.DROPZONE:
                            tile = self.tiles[f'drone_{air_pos}_{drone_orientation}_dropzone']
                        else:
                            tile = self.tiles[f'drone_{air_pos}_{drone_orientation}']

                    # draw charging bar
                    charge_bar = Image.new('RGB', (10, 2), (0, 0, 0))
                    draw = ImageDraw.Draw(charge_bar)
                    charge_level = int(charge[air_pos]) // 10
                    draw.rectangle([(0, 0), (charge_level, 1)], fill='green')
                    if self.trace_length > 0:
                        if len(self.traces[air_pos]) >= self.trace_length:
                            ri, rj = self.traces[air_pos][0]
                            self.trace_paths[ri][rj] = None
                        self.traces[air_pos].append((i, j))
                        self.trace_paths[i][j] = air_pos
                elif has_ground_pos:
                    # has_ground_pos
                    if ground_pos == Object.PACKET:
                        tile = self.tiles['packet']
                    elif ground_pos == Object.DROPZONE:
                        tile = self.tiles['dropzone']
                    elif ground_pos == Object.STATION:
                        tile = self.tiles['station']
                    elif ground_pos == Object.SKYSCRAPER:
                        tile = self.tiles['skyscraper']
                    else:
                        raise ValueError(f'Unexpected ground pos value {ground_pos}')
                else:
                    # empty tile
                    tile = Image.new('RGB', (16, 16), (0, 0, 0))

                # Paste tile on frame
                draw = ImageDraw.Draw(tile)
                if self.trace_paths[i][j] is not None and self.trace_paths[i][j] in self.trace_drone_ids_only:
                    # TODO: this is still to improve
                    # ship_color = self.ship_colors[self.trace_paths[i][j]]
                    draw.rectangle((0, 0, 15, 15), outline=(127, 127, 255), width=1)
                else:
                    draw.rectangle((0, 0, 15, 15), outline='black', width=1)

                tile_x = j * self.tiles_size + (j + 1) * self.render_padding
                tile_y = i * self.tiles_size + (i + 1) * self.render_padding
                frame.paste(tile, (tile_x, tile_y))

                if charge_bar:
                    frame.paste(charge_bar, (tile_x + 2, tile_y + self.tiles_size + 2))

        # generate metric panel
        metric_panel = copy.copy(self.metric_panel)
        draw_handle = ImageDraw.Draw(metric_panel, mode='RGB')
        number_indent = 6
        draw_handle.text((self.render_padding + 2, self.render_padding), f'Step: {step:>{number_indent},}', fill='black', font=self.font)
        draw_handle.text((self.render_padding + 2, self.render_padding + self.line_spacing), 'Rewards', fill='black', font=self.font)
        for player_id in range(len(rewards)):
            draw_handle.text((self.render_padding + 2, self.render_padding + self.line_spacing * (2 + player_id)), f'P{player_id:}: {rewards[player_id]:>{number_indent + 2}.1f}', fill='black', font=self.font)

        frame = np.vstack([np.hstack([frame, np.vstack([self.panel, metric_panel])]), self.legend])
        frame = Image.fromarray(frame)

        # Rescale frame
        rescale = lambda old_size: int(old_size * self.rgb_render_rescale)
        frame = frame.resize(size=(rescale(frame.size[0]), rescale(frame.size[1])), resample=Image.NEAREST)
        return frame

    def save_frame(self, img: Image.Image, step: int, output_dir: str):
        img.save(os.path.join(output_dir, f'{step:04d}.{self.image_format}'))

    def make_subprocess_call(self, command: str, shell: bool = False):
        result = subprocess.run(
            command.split(),
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout = result.stdout.decode('utf-8')
        stderr = result.stderr.decode('utf-8')
        return result.returncode, stdout, stderr

    def generate_video(
            self,
            input_folder_path: str,
            output_path: str,
            output_resolution: Tuple[int, int] = (600, 600),
            ffmpeg_exec: str = 'ffmpeg',
            fps: int = 4):
        # Generate Normal Sized Video
        frames_path = os.path.join(input_folder_path, f"%04d.{self.image_format}")
        res = f'{output_resolution[0]}x{output_resolution[1]}'
        return_code, output, output_err = self.make_subprocess_call(
            ffmpeg_exec +
            f" -y -r {fps} -start_number 0 -i " +
            frames_path +
            f" -c:v libx264 -vf fps={fps} -pix_fmt yuv420p -s {res} " +
            output_path
        )
        if return_code != 0:
            raise Exception(output_err)
        return output_path


if __name__ == "__main__":
    from env import DroneEnvParams, DeliveryDrones
    import jax
    import jax.random
    import jax.numpy as jnp

    grid_size = 10
    n_drones = 3
    drone_density = n_drones / (grid_size ** 2)
    print(f'Num drones: {n_drones}, grid: {grid_size}x{grid_size}, drone density: {drone_density:.2f}')

    # params = DroneEnvParams(n_drones=n_drones, grid_size=grid_size, packets_factor=1, dropzones_factor=1, stations_factor=1)
    params = DroneEnvParams(n_drones=n_drones, grid_size=grid_size)
    env = DeliveryDrones()
    rng = jax.random.PRNGKey(0)
    num_steps = 200
    state = env.reset(rng, params)
    step_jit = jax.jit(env.step, static_argnums=(3,))
    renderer = Renderer(params.n_drones, params.grid_size, rgb_render_rescale=4)
    renderer.init()

    # starting state
    img = renderer.render_frame(*convert_jax_state(
        0, state, jnp.array(params.n_drones * [4]), jnp.array(params.n_drones * [0.0])))
    img.save(f'output/0000.png')

    for step in range(1, num_steps):
        rng, key = jax.random.split(rng)
        actions = jax.random.randint(key, (params.n_drones,), 0, 5, dtype=jnp.int32)
        state, rewards, dones = step_jit(rng, state, actions, params)

        img = renderer.render_frame(*convert_jax_state(step, state, actions, rewards))
        renderer.save_frame(img, step, 'output')
        print('step', step)
        print(env.format_action(*actions), dones, state.carrying_package)
        print('x:', state.air_x, 'y:', state.air_y)
        print(renderer.orientation)

    renderer.generate_video('output', 'out.mp4', output_resolution=img.size, fps=3)
