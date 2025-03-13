import os
import numpy as np
import random
import torch


def sample_waypoints_from_graph(waypoint_graph):
    current_node = waypoint_graph[0]
    waypoints = [list(current_node.point)]
    while len(current_node.next_node_ids) > 0:
        next_node_id = random.choices(current_node.next_node_ids, weights=current_node.next_edges, k=1)[0]
        current_node = waypoint_graph[next_node_id]
        waypoints.append(list(current_node.point))
#    print("waypoints")
#    print(waypoints)
    return waypoints


def save_video(imgs, filename, batch_index=0, fps=10, web_browser_friendly=False):
    import cv2
    img_stack = [cv2.cvtColor(
        img[batch_index].cpu().numpy().astype(
            np.uint8).transpose(1, 2, 0), cv2.COLOR_RGB2BGR
    )
        for img in imgs]

    w = img_stack[0].shape[0]
    h = img_stack[0].shape[1]
    output_format = cv2.VideoWriter_fourcc(*'mp4v')

    vid_out = cv2.VideoWriter(filename=filename,
                              fourcc=output_format,
                              fps=fps,
                              frameSize=(w, h))

    for frame in img_stack:
        vid_out.write(frame)

    vid_out.release()

    if web_browser_friendly:
        import uuid
        temp_filename = os.path.join(os.path.dirname(
            filename), str(uuid.uuid4()) + '.mp4')
        os.rename(filename, temp_filename)
        os.system(
            f"ffmpeg -y -i {temp_filename} -hide_banner -loglevel error -vcodec libx264 -f mp4 {filename}")
        os.remove(temp_filename)


def convert_to_json(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert to list
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert to list
    if isinstance(obj, dict):
        return {k: convert_to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json(i) for i in obj]
    if isinstance(obj, tuple):
        return tuple(convert_to_json(i) for i in obj)
    return obj


def set_seeds(seed, logger=None):
    if seed is None:
        seed = np.random.randint(low=0, high=2**32 - 1)
    if logger is not None:
        logger.info(f"seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed
