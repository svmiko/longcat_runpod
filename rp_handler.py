import os
import json
import base64
import subprocess
import time
import glob
import shutil
import runpod


def handler(job):
    job_input = job["input"]

    # Get parameters
    prompt = job_input.get("prompt", "A person talking")
    audio_base64 = job_input.get("audio")  # Base64 encoded audio (required)
    image_base64 = job_input.get("image")  # Base64 encoded image (optional)

    # Script parameters
    resolution = job_input.get("resolution", "480p")
    num_inference_steps = job_input.get("num_inference_steps", 16)
    text_guidance_scale = job_input.get("text_guidance_scale", 1.0)
    audio_guidance_scale = job_input.get("audio_guidance_scale", 1.0)
    num_segments = job_input.get("num_segments", 0)  # 0 = auto-calculate
    stage_1 = job_input.get("stage_1", "ai2v" if image_base64 else "at2v")
    ref_img_index = job_input.get("ref_img_index", 10)
    mask_frame_range = job_input.get("mask_frame_range", 3)

    # Create temp directory for this job
    job_id = job.get("id", str(int(time.time())))
    temp_dir = f"/tmp/job_{job_id}"
    os.makedirs(temp_dir, exist_ok=True)

    # Save audio to temp file
    audio_path = os.path.join(temp_dir, "input_audio.wav")
    with open(audio_path, "wb") as f:
        f.write(base64.b64decode(audio_base64))

    # Save image if provided
    image_path = None
    if image_base64:
        image_path = os.path.join(temp_dir, "input_image.png")
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_base64))

    # Create input JSON
    input_json = {
        "prompt": prompt,
        "cond_audio": {
            "person1": audio_path
        }
    }
    if image_path:
        input_json["cond_image"] = image_path

    input_json_path = os.path.join(temp_dir, "input.json")
    with open(input_json_path, "w") as f:
        json.dump(input_json, f)

    # Output directory
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Build command
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "run_demo_avatar_single_audio_to_video.py",
        f"--input_json={input_json_path}",
        f"--output_dir={output_dir}",
        f"--checkpoint_dir=./weights/LongCat-Video-Avatar",
        f"--resolution={resolution}",
        f"--num_inference_steps={num_inference_steps}",
        f"--text_guidance_scale={text_guidance_scale}",
        f"--audio_guidance_scale={audio_guidance_scale}",
        f"--num_segments={num_segments}",
        f"--stage_1={stage_1}",
        f"--ref_img_index={ref_img_index}",
        f"--mask_frame_range={mask_frame_range}",
    ]

    print(f"Running command: {' '.join(cmd)}")

    # Run the script
    try:
        result = subprocess.run(
            cmd,
            cwd="/workspace/LongCat-Video",
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

        if result.returncode != 0:
            return {"error": f"Script failed with return code {result.returncode}", "stderr": result.stderr}

    except subprocess.TimeoutExpired:
        return {"error": "Script timed out after 1 hour"}
    except Exception as e:
        return {"error": str(e)}

    # Find the output video (get the latest mp4 file)
    video_files = glob.glob(os.path.join(output_dir, "*.mp4"))
    if not video_files:
        return {"error": "No output video found", "output_dir_contents": os.listdir(output_dir)}

    # Get the latest video file
    latest_video = max(video_files, key=os.path.getmtime)
    print(f"Output video: {latest_video}")

    # Read and encode video
    with open(latest_video, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')

    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "video": video_base64,
        "video_filename": os.path.basename(latest_video)
    }


# Start RunPod handler
runpod.serverless.start({"handler": handler})
