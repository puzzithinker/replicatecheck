import replicate
import asyncio
from concurrent.futures import ThreadPoolExecutor

## 需要提前运行:
### pip install -r requirements.txt
### export REPLICATE_API_TOKEN=xxxxxxxx

def run_model_sync():
    items = [
    "liuyifei, wearing a beanie, sits at a cafe table holding a warm coffee cup.",
    "liuyifei builds a chair, surrounded by tools and wood in a workshop.",
    "A smiling liuyifei looks directly at the camera for a portrait.",
    "liuyifei hikes through a forest trail, carrying a backpack.",
    "liuyifei prepares a meal in a kitchen, chopping vegetables.",
    "liuyifei plays guitar on a stage, illuminated by spotlights.",
    "liuyifei reads a book while relaxing on a park bench.",
    "liuyifei paints a canvas with a focused expression.",
    "liuyifei rides a bicycle along a scenic coastal road.",
    "liuyifei works on a laptop at a desk in a modern office."
]
    for line in items:
        result = replicate.run(
            "zgimszhd61/flux-dev-lora-trainer:129cd7f38a495f4550bb146da6d8f7fcd172d906cfcd799a17241a5c00d39756",
            input={
                "prompt":line,
                "model": "dev",
                "lora_scale": 1,
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "png",
                "guidance_scale": 3.5,
                "output_quality": 90,
                "prompt_strength": 0.8,
                "extra_lora_scale": 1,
                "num_inference_steps": 28,
                "disable_safety_checker": True
            }
        )
        print(result)

    return "Everything is done..."

async def run_model():
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, run_model_sync)
    return result

async def main():
    print("模型正在运行，稍后会返回结果...")
    result = await run_model()  # 启动模型运行并等待结果
    print("模型运行完成，结果如下：")
    print(result)  # 打印模型输出的结果

asyncio.run(main())