import os
import random
import numpy as np
import torch
import hashlib
from slack_sdk import WebClient
from datetime import datetime
import time


def calculate_file_hash(file_path, algorithm="md5"):
    hash_object = hashlib.new(algorithm)
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096 * 4096), b""):
            hash_object.update(chunk)
    return hash_object.hexdigest()


def has_overlap(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1 & set2
    return len(intersection) > 0


class SlackNotify:
    def __init__(self, token, channel, userid, init_text):
        self.client = WebClient(token=token)
        self.channel = channel
        self.userid = userid
        self.parent_ts = None
        self.send_parent_message(init_text)

    def send_parent_message(self, init_text):
        self.client.chat_postMessage(channel=self.channel, text=init_text, thread_ts=None)
        time.sleep(5)
        self.parent_ts = self.get_parent_ts()

    def send_reply(self, text, mention=False):
        if mention:
            text = f"<@{self.userid}>\n" + text
        self.client.chat_postMessage(channel=self.channel, text=text, thread_ts=self.parent_ts)

    def get_parent_ts(self):
        response = self.client.conversations_history(channel=self.channel, latest=int(datetime.now().timestamp()), inclusive=True, limit=1)
        ts = response["messages"][0]["ts"]
        return ts


def set_seed(seed=42, cudnn_deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
