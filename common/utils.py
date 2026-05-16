import json
import re
import os
import random
import numpy as np

import multiprocessing
from typing import Tuple
from statistics import mean
import openai
from openai import OpenAI,AsyncOpenAI

# API configuration - please set your own API endpoint and key
API_URL = "YOUR_API_URL_HERE"
API_KEY = "YOUR_API_KEY_HERE"
MODEL_NAME = "gpt-4o-mini"
client = OpenAI(base_url=API_URL,
                       api_key=API_KEY,
                       )



def read_json(file_path):
    assert str(file_path).endswith(".json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(js_obj, file_path):
    assert str(file_path).endswith(".json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(js_obj, f, indent=4)


def read_txt(file_path):
    assert str(file_path).endswith(".txt")
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data
