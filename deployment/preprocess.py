import os
import ffmpeg


def sample_video_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return (
        ffmpeg
            .input(video_path)
            # .filter('select', "eq(n, 10000)")
            .filter('select', 'not(mod(n,15))')
            .output(os.path.join(output_dir, "out%d.jpg"),  **{ "vsync": "vfr", 'qscale:v': 1})
            .run()
    )
