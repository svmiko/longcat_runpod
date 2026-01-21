import os
import json
import base64
import subprocess
import time
import glob
import shutil
import runpod


def build_error_response(error_msg, error_code, job_id, metadata, start_time, **extra):
    """Build a structured error response for webhook consumption."""
    response = {
        "status": "failed",
        "error": error_msg,
        "error_code": error_code,
        "runpod_job_id": job_id,
        "timing": {
            "total_seconds": round(time.time() - start_time, 2)
        }
    }
    if metadata:
        response["metadata"] = metadata
    response.update(extra)
    return response


def build_success_response(video_base64, video_filename, job_id, metadata, start_time, generation_time):
    """Build a structured success response for webhook consumption."""
    response = {
        "status": "completed",
        "video": video_base64,
        "video_filename": video_filename,
        "runpod_job_id": job_id,
        "timing": {
            "generation_seconds": round(generation_time, 2),
            "total_seconds": round(time.time() - start_time, 2)
        }
    }
    if metadata:
        response["metadata"] = metadata
    return response


def handler(job):
    """
    RunPod serverless handler for LongCat video generation.

    Supports webhooks: When a job is submitted with a webhook URL,
    RunPod will POST this handler's return value to that URL.

    Input:
        audio (str): Base64 encoded audio (required)
        image (str): Base64 encoded image (optional)
        prompt (str): Text prompt (optional)
        metadata (dict): Passthrough metadata returned in output (optional)
            - Useful for tracking job_id, session_id, slide_index, etc.
        ... other generation parameters

    Output (on success):
        status: "completed"
        video: Base64 encoded MP4
        video_filename: Output filename
        metadata: Passthrough from input (if provided)
        timing: Execution timing info
        runpod_job_id: The RunPod job ID

    Output (on error):
        status: "failed"
        error: Error message
        error_code: Error type identifier
        metadata: Passthrough from input (if provided)
        runpod_job_id: The RunPod job ID
    """
    start_time = time.time()

    # Debug: print current directory and list files
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in /app: {os.listdir('/app') if os.path.exists('/app') else 'NOT FOUND'}")
    print(f"Files in /workspace: {os.listdir('/workspace') if os.path.exists('/workspace') else 'NOT FOUND'}")
    print(f"Files in /runpod-volume: {os.listdir('/runpod-volume') if os.path.exists('/runpod-volume') else 'NOT FOUND'}")
    print(f"Weights at /workspace/weights: {os.listdir('/workspace/weights') if os.path.exists('/workspace/weights') else 'NOT FOUND'}")
    print(f"Weights at /runpod-volume/weights: {os.listdir('/runpod-volume/weights') if os.path.exists('/runpod-volume/weights') else 'NOT FOUND'}")
    print(f"LongCat-Video exists at /workspace/weights: {os.path.exists('/workspace/weights/LongCat-Video')}")
    print(f"LongCat-Video-Avatar exists at /workspace/weights: {os.path.exists('/workspace/weights/LongCat-Video-Avatar')}")

    job_input = job["input"]
    job_id = job.get("id", str(int(time.time())))

    # Extract passthrough metadata (returned unchanged in output)
    metadata = job_input.get("metadata", {})

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
        f"--checkpoint_dir=/runpod-volume/weights/LongCat-Video-Avatar",
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
    generation_start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd="/app",
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        generation_time = time.time() - generation_start
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

        if result.returncode != 0:
            return build_error_response(
                error_msg=f"Script failed with return code {result.returncode}",
                error_code="GENERATION_FAILED",
                job_id=job_id,
                metadata=metadata,
                start_time=start_time,
                stderr=result.stderr
            )

    except subprocess.TimeoutExpired:
        return build_error_response(
            error_msg="Script timed out after 1 hour",
            error_code="TIMEOUT",
            job_id=job_id,
            metadata=metadata,
            start_time=start_time
        )
    except Exception as e:
        return build_error_response(
            error_msg=str(e),
            error_code="UNEXPECTED_ERROR",
            job_id=job_id,
            metadata=metadata,
            start_time=start_time
        )

    # Find the output video (get the latest mp4 file)
    video_files = glob.glob(os.path.join(output_dir, "*.mp4"))
    if not video_files:
        return build_error_response(
            error_msg="No output video found",
            error_code="NO_OUTPUT",
            job_id=job_id,
            metadata=metadata,
            start_time=start_time,
            output_dir_contents=os.listdir(output_dir)
        )

    # Get the latest video file
    latest_video = max(video_files, key=os.path.getmtime)
    print(f"Output video: {latest_video}")

    # Read and encode video
    with open(latest_video, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')

    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    return build_success_response(
        video_base64=video_base64,
        video_filename=os.path.basename(latest_video),
        job_id=job_id,
        metadata=metadata,
        start_time=start_time,
        generation_time=generation_time
    )


# Start RunPod handler
runpod.serverless.start({"handler": handler})
