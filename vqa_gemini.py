import argparse
import os
import re
import json
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold

genai.configure(api_key=os.environ["API_KEY"])

def answer_gemini(video, qa_prompt, model):
    '''
    Get an answer from Gemini API.

    Parameters:
    video - filename of the video
    qa_prompt - question to ask about the video
    model - gemini model object

    Returns:
    initial_answer - answer from Gemini
    final_answer - answer after additional object prompt
    '''

    try:
        video_file = genai.get_file(video)
    except:
        video_file = genai.upload_file(path=video)
        while True:
            while video_file.state.name =="PROCESSING":
                time.sleep(5)
                video_file = genai.get_file(video_file.name)
            if video_file.state.name == "ACTIVE":
                break
            if video_file.state.name == "FAILED":
                video_file = genai.upload_file(path=video)

    response = model.generate_content([video_file,qa_prompt])
    time.sleep(5)
    initial_answer = response.text

    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [qa_prompt],
            },
            {
                "role": "model",
                "parts": [initial_answer],
            },
        ]
    )

    time.sleep(5)
    response = chat_session.send_message(
        "Return only the object name corresponding to the answer."
    )

    final_answer = response.text

    return initial_answer, final_answer


def main(video_folder, metadata_file, model_name, num_frame):

    # Parameters for text generation
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 2048,
        "response_mime_type": "text/plain",
    }
    
    # Safety thresholds for Gemini to bypass
    safety_settings = [
        {
            "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            "threshold": HarmBlockThreshold.BLOCK_NONE
        }, 
        {
            "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            "threshold": HarmBlockThreshold.BLOCK_NONE
        },
        {
            "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
            "threshold": HarmBlockThreshold.BLOCK_NONE
        },
        {
            "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            "threshold": HarmBlockThreshold.BLOCK_NONE
        }
    ]

    # Gemini model object
    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="You are a helpful assistant helping the user process descriptive logs of a video to answer some questions.",
    )


    video_folder = Path(video_folder)
    metadata_file = Path(metadata_file)
    split = 'val'
    output_file = f"vqa_{split}_{model_name.split('/')[-1]}_output.json"
    try:
        with open(output_file, 'r') as f:
            output_json = json.load(f)
        results = output_json.copy()
    except:
        results = {}

    # Read metadata JSON file
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    for video_id, video_data in tqdm(list(metadata.items()), desc="Processing videos", unit="video"):
        video_file = video_folder / f"{video_id}.mp4"

        if not video_file.exists():
            tqdm.write(f"Video file not found: {video_file}")
            continue

        video_results = []

        # If video has already been processed, skip
        if video_id in results.keys():
            if results[video_id][0]['model_output'] != 'cup':
                continue
            
        # Process each question for the video
        for q_num, question in enumerate(video_data['grounded_question']):

            result = "cup"
            input_question = question['question'].replace('Track', 'What is/are')

            try:
                _, result = answer_gemini(video_file, input_question, model)
                result = result.strip()
            except Exception as e:
                tqdm.write(f"Error processing {video_file}: {str(e)}")
     
            video_results.append({'original_question': question['question'], 'input_question': input_question, 'model_output': result})

        results[video_id] = video_results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Write results to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    video_folder = ''
    metadata_file = f'{video_folder[:-5]}/grounded_question_valid.json'
    model = "gemini-1.5-flash"
    num_frames = 16
    main(video_folder, metadata_file, model, num_frames)
