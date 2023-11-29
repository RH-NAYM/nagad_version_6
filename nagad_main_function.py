import torch
import asyncio
import json
import pandas as pd
from datetime import datetime
# from Data.NBRTU_Data import nbrtuModel, old_nbrtuModel, NBRTU_val, ndel_items, nagad_items, bkash_items, rocket_items, tap_items, upay_items, new_items
from Data.NBRTU_Data import nbrtuModel, NBRTU_val, ndel_items, nagad_items, bkash_items, rocket_items, tap_items, upay_items
import pytz
import requests
import cv2
import numpy as np


def get_bd_time():
    bd_timezone = pytz.timezone("Asia/Dhaka")
    time_now = datetime.now(bd_timezone)
    current_time = time_now.strftime("%I:%M:%S %p")
    return current_time

# Check Blurry
def assess_image_quality(image_url):
    try:
        resp = requests.get(image_url, stream=True)
        resp.raise_for_status()

        # Read the image
        image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Assess image blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_value < 100:
            return "Blurry"
        else:
            return "Not blurry"

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    


# Nagad Split Function
async def process_nagad_item(nagad_item, nbrtuDict, nagad):
    if nagad_item in nbrtuDict:
        n = {nagad_item: nbrtuDict[nagad_item]}
        nagad.update(n)

# Bkash Split Function
async def process_bkash_item(bkash_item, nbrtuDict, bkash):
    if bkash_item in nbrtuDict:
        b = {bkash_item: nbrtuDict[bkash_item]}
        bkash.update(b)

# Rocket Split Function
async def process_rocket_item(rocket_item, nbrtuDict, rocket):
    if rocket_item in nbrtuDict:
        r = {rocket_item: nbrtuDict[rocket_item]}
        rocket.update(r)

# Tap Split Function
async def process_tap_item(tap_item, nbrtuDict, tap):
    if tap_item in nbrtuDict:
        t = {tap_item: nbrtuDict[tap_item]}
        tap.update(t)

# Upay Split Function
async def process_upay_item(upay_item, nbrtuDict, upay):
    if upay_item in nbrtuDict:
        u = {upay_item: nbrtuDict[upay_item]}
        upay.update(u)


# Object Detection (Main Function)
async def detect_objects(model, url):
    result = await asyncio.get_event_loop().run_in_executor(None, model, url)
    result = result.pandas().xyxy[0].sort_values(by=['xmin', 'ymax'])
    df = pd.DataFrame(result)
    name_counts = df.groupby('name').size().to_dict()
    result_dict = {}
    for index, row in df.iterrows():
        name = row['name']
        result_dict[name] = name_counts.get(name, 0)
    return result_dict

# Multi-Threading detection 
async def detect_sequence(url):

    nbrtuModel.conf = 0.55
    nbrtuModel.iou = 0.25
    # tasks = [detect_objects(nbrtuModel, url)]
    # results = await asyncio.gather(*tasks)
    # tasks = [
    #     detect_objects(old_nbrtuModel, url),
    #     detect_objects(nbrtuModel, url)
    # ]
    # results = await asyncio.gather(*tasks)

    # old, new = results
    results = await asyncio.create_task(detect_objects(nbrtuModel,url))
    nbrtuDict = results
    # nbrtuDict = old
    # for item in new_items:
    #     if item in new:
    #         data = {item:new[item]}
    #         nbrtuDict.update(data)
        

    # Nagad Bkash Rocket Tap Upay Validation :
    for val_item in NBRTU_val:
        if val_item in nbrtuDict and nbrtuDict[val_item] > 0:
            nbrtu_validation_single = {val_item: "yes"}
            nbrtuDict.update(nbrtu_validation_single)


    # Remove Extra Items : 
    for nagad_remove_item in ndel_items:
        if nagad_remove_item in nbrtuDict:
            del nbrtuDict[nagad_remove_item]

    nagad = {}
    bkash = {}
    rocket = {}
    tap = {}
    upay = {}

    # Using asyncio.gather to await multiple process functions concurrently
    process_nagad_tasks = [process_nagad_item(nagad_item, nbrtuDict, nagad) for nagad_item in nagad_items]
    process_bkash_tasks = [process_bkash_item(bkash_item, nbrtuDict, bkash) for bkash_item in bkash_items]
    process_rocket_tasks = [process_rocket_item(rocket_item, nbrtuDict, rocket) for rocket_item in rocket_items]
    process_tap_tasks = [process_tap_item(tap_item, nbrtuDict, tap) for tap_item in tap_items]
    process_upay_tasks = [process_upay_item(upay_item, nbrtuDict, upay) for upay_item in upay_items]

    await asyncio.gather(*process_nagad_tasks, *process_bkash_tasks, *process_rocket_tasks, *process_tap_tasks, *process_upay_tasks)

    nagad_detection = {
        'nagad': nagad,
        'bkash': bkash,
        'rocket': rocket,
        'tap': tap,
        'upay': upay
    }


    nagad_result = json.dumps(nagad_detection)
    return nagad_result

async def mainDetect(url):
    try:
        result = await detect_sequence(url)
        return result
    finally:
        torch.cuda.empty_cache()
        pass
