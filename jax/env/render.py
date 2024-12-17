from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Dict, Tuple, Literal
import subprocess
import itertools
import copy
import numpy as np
import os


assets_path = os.path.dirname(os.path.realpath(__file__))
font_path = os.path.join(assets_path, "assets", "font", "Inconsolata-Bold.ttf")
sprite_path = os.path.join(assets_path, "assets", "16ShipCollection.png")

OBJ_SKYSCRAPER = 2
OBJ_STATION = 3
OBJ_DROPZONE = 4
OBJ_PACKET = 5


class Renderer:
    def __init__(
            self,
            n_drones: int,
            grid_size: int,
            player_name_mappings: Optional[Dict[int, str]] = None,
            rgb_render_rescale: float = 1.0):
        self.n_drones = n_drones
        self.grid_size = grid_size
        self.player_name_mappings = player_name_mappings
        self.rgb_render_rescale = rgb_render_rescale
        self.orientation = self.n_drones * ['right']

    def init(self):
        # Load RGBA image
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
            # offset_px = {'right': (offset, 0), 'up': (0, offset), 'left': (offset, 0), 'down': (0, offset)}[orientation]
            # overlay = Image.new('RGBA', [img_a.size[0] + offset_px[0], img_a.size[1] + offset_px[1]])
            overlay = Image.new('RGBA', [img_a.size[0] + offset, img_a.size[1] + offset])
            if orientation == 'right':
                overlay.paste(img_a, (offset, 0), img_a)  # move package to the right
                overlay.paste(img_b, (0, 0), img_b)
            elif orientation == 'down':
                overlay.paste(img_a, (0, offset), img_a)  # move package down
                overlay.paste(img_b, (0, 0), img_b)
            elif orientation == 'up':
                overlay.paste(img_a, (0, 0), img_a)
                overlay.paste(img_b, (0, offset), img_b)  # move drone down
            else:
                # left
                overlay.paste(img_a, (0, 0), img_a)
                overlay.paste(img_b, (offset, 0), img_b)  # move drone to right
            return overlay

        for drone_idx in range(self.n_drones):
            i, j = next(ships_iter)
            tile = get_ships_tile(i, j)
            for rot, orientation in zip([0, 90, 180, 270], ['right', 'up', 'left', 'down']):
                label = f'drone_{drone_idx}_{orientation}'
                tile_rot = copy.copy(tile).rotate(rot)  # counter-clock wise
                self.tiles[label] = tile_rot
                self.tiles[label + '_packet'] = overlay(self.tiles['packet'], tile_rot, orientation)
                self.tiles[label + '_charging'] = overlay(self.tiles['station'], tile_rot, orientation, offset=0)
                self.tiles[label + '_dropzone'] = overlay(self.tiles['dropzone'], tile_rot, orientation, offset=0)

        # Create empty frame
        self.render_padding, self.tiles_size = 10, 16
        frames_size = self.tiles_size * self.grid_size + self.render_padding * (self.grid_size + 1)
        self.empty_frame = np.full(shape=(frames_size, frames_size, 4), fill_value=0, dtype=np.uint8)
        self.empty_frame[:, :, 3] = 255  # Remove background transparency

        # Side panel
        background_color = (20, 200, 200)
        self.panel = Image.new('RGBA', (120, self.empty_frame.shape[1]), color=background_color)
        draw_handle = ImageDraw.Draw(self.panel, mode='RGBA')
        font = ImageFont.truetype(font_path, 16)

        for drone_idx in range(self.n_drones):
            # Print sprite
            drone_sprite = self.tiles['drone_{}_right'.format(drone_idx)]
            sprite_x = self.render_padding
            sprite_y = drone_idx * self.tiles_size + (drone_idx + 1) * self.render_padding
            self.panel.paste(drone_sprite, (sprite_x, sprite_y), drone_sprite)

            player_name = 'Player {:>2}'.format(drone_idx)
            if self.player_name_mappings is not None:
                # Optional setting for rendering videos on AIcrowd evaluator
                player_name = self.player_name_mappings.get(drone_idx, '')

            # # Print text
            text_x = sprite_x + self.tiles_size + self.render_padding
            text_y = sprite_y - 1
            text_color = (0, 0, 0)
            draw_handle.text((text_x, text_y), player_name, fill=text_color, font=font)


    def render(self, ground: np.ndarray, air: np.ndarray, carrying_package: np.ndarray, charge: np.ndarray, actions: np.ndarray):
        # Render frame
        frame = Image.fromarray(self.empty_frame.copy())
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Check tile
                ground_pos = ground[i, j]
                air_pos = air[i, j]
                has_ground_pos = (ground_pos != 0) and (ground_pos is not None)
                has_air_pos = air_pos is not None
                charge_bar = None
                if not has_air_pos and not has_ground_pos:
                    continue
                elif has_air_pos:
                    drone_orientation = {0: 'left', 1: 'down', 2: 'right', 3: 'up', 4: self.orientation[air_pos]}[actions[air_pos]]
                    self.orientation[air_pos] = drone_orientation
                    if carrying_package[air_pos]:
                        tile = self.tiles[f'drone_{air_pos}_{drone_orientation}_packet']
                    else:
                        if ground_pos == OBJ_STATION:
                            tile = self.tiles[f'drone_{air_pos}_{drone_orientation}_charging']
                        elif ground_pos == OBJ_DROPZONE:
                            tile = self.tiles[f'drone_{air_pos}_{drone_orientation}_dropzone']
                        else:
                            tile = self.tiles[f'drone_{air_pos}_{drone_orientation}']

                    # draw charging bar
                    charge_bar = Image.new('RGBA', (10, 2), (0, 0, 0, 0))  # Transparent background
                    draw = ImageDraw.Draw(charge_bar)
                    charge_level = int(charge[air_pos]) // 10
                    draw.rectangle([(0, 0), (charge_level, 1)], fill='green')
                else:
                    # has_ground_pos
                    if ground_pos == OBJ_PACKET:
                        tile = self.tiles['packet']
                    elif ground_pos == OBJ_DROPZONE:
                        tile = self.tiles['dropzone']
                    elif ground_pos == OBJ_STATION:
                        tile = self.tiles['station']
                    elif ground_pos == OBJ_SKYSCRAPER:
                        tile = self.tiles['skyscraper']
                    else:
                        raise ValueError(f'Unexpected ground pos value {ground_pos}')

                # Paste tile on frame
                tile_x = j * self.tiles_size + (j + 1) * self.render_padding
                tile_y = i * self.tiles_size + (i + 1) * self.render_padding
                frame.paste(tile, (tile_x, tile_y), mask=tile)

                if charge_bar:
                    frame.paste(charge_bar, (tile_x + 2, tile_y + self.tiles_size + 2), mask=charge_bar)


        frame = Image.fromarray(np.hstack([frame, self.panel]))

        # Rescale frame
        rescale = lambda old_size: int(old_size * self.rgb_render_rescale)
        frame = frame.resize(size=(rescale(frame.size[0]), rescale(frame.size[1])), resample=Image.NEAREST)
        return frame

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
            input_format: Literal['jpg', 'png'] = 'png',
            fps: int = 4):
        # Generate Normal Sized Video
        frames_path = os.path.join(input_folder_path, f"%04d.{input_format}")
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



def convert_jax_state(state, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ground = jax.device_get(state.ground)
    actions = jax.device_get(actions)
    charge = jax.device_get(state.charge)
    air_x = jax.device_get(state.air_x)
    air_y = jax.device_get(state.air_y)
    air = np.zeros_like(ground, dtype=np.object_)
    air[:] = None
    air[air_y, air_x] = np.arange(air_x.size)
    carrying_package = jax.device_get(state.carrying_package)
    return ground, air, carrying_package, charge, actions


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
    img = renderer.render(*convert_jax_state(state, jnp.array(params.n_drones * [4])))
    img.save(f'output/0000.png')

    for step in range(1, num_steps):
        rng, key = jax.random.split(rng)
        actions = jax.random.randint(key, (params.n_drones,), 0, 5, dtype=jnp.int32)
        state, rewards, dones = step_jit(rng, state, actions, params)

        img = renderer.render(*convert_jax_state(state, actions))
        img.save(f'output/{step:04d}.png')
        print('step', step)
        print(env.format_action(*actions), dones, state.carrying_package)
        print('x:', state.air_x, 'y:', state.air_y)
        print(renderer.orientation)

    renderer.generate_video('output', 'out.mp4', output_resolution=img.size, input_format='png', fps=3)
