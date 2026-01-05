import os, json, time, sys, asyncio
import requests
from typing import Any, List, Dict, Literal, Union, Optional
from pydantic import BaseModel, ValidationError
from abc import ABC, abstractmethod
from enum import Enum

MOCKS = {}
ANALYSIS_MODE = False
_ANALYSIS_RESULTS = None
