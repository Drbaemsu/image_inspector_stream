import nest_asyncio
nest_asyncio.apply()

import playwright
playwright.install()

import os
from playwright.async_api import async_playwright
from PIL import Image
from io import BytesIO
import aiohttp
from bs4 import BeautifulSoup
import urllib.parse
import cv2
import numpy as np
import streamlit as st
import asyncio
import re
import pandas as pd

# 이미지 저장 경로 설정
images_folder = 'naver_map_images'
os.makedirs(images_folder, exist_ok=True)

# 코드 생략 (기존 코드와 동일)

def streamlit_app():
    st.title("Image Similarity Finder")
    st.write("Upload an image and a file containing URLs (one per line) to find similar images from the downloaded sets.")

    image = st.file_uploader("Upload an image", type=["jpg", "png"])
    urls_file = st.file_uploader("Upload URLs file", type=["txt"])

    if image and urls_file:
        try:
            image = Image.open(image)
            urls_file = urls_file.getvalue().decode("utf-8")
            result = asyncio.run(run_download_and_compare(image, urls_file))

            df = pd.DataFrame(result, columns=["URL", "유사한 이미지 여부"])
            st.dataframe(df)

            csv_file = "results.csv"
            df.to_csv(csv_file, index=False)
            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False),
                file_name=csv_file,
                mime="text/csv",
            )
        except Exception as e:
            st.error(str(e))

if __name__ == "__main__":
    streamlit_app()
