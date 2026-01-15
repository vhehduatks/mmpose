import os
import argparse
import json
from PIL import Image
from tqdm import tqdm


def adjust_coordinates(points, crop_x_start, crop_y_start, crop_x_end, crop_y_end, new_width, new_height):
	adjusted_points = [[], []]
	for x, y in zip(points[0], points[1]):
		# 크롭된 좌표로 변환
		new_x = (x - crop_x_start) / (crop_x_end - crop_x_start) * new_width
		new_y = (y - crop_y_start) / (crop_y_end - crop_y_start) * new_height
		adjusted_points[0].append(new_x)
		adjusted_points[1].append(new_y)
	
	return adjusted_points

def process_json(json_file):
	# JSON 파일 읽기
	with open(json_file, 'r') as file:
		data = json.load(file)

	# 원본 이미지 크기 및 크롭된 영역 정의
	original_width, original_height = 1280, 800
	crop_x_start, crop_y_start = 200, 0
	crop_x_end, crop_y_end = original_width - 165, original_height
	new_width, new_height = 256, 256

	# pts2d_fisheye 조정
	if 'pts2d_fisheye' in data:
		data['pts2d_fisheye'] = adjust_coordinates(data['pts2d_fisheye'], crop_x_start, crop_y_start, crop_x_end, crop_y_end, new_width, new_height)

	# 수정된 JSON 파일을 원본 파일에 저장
	with open(json_file, 'w') as file:
		json.dump(data, file, indent=4)


def process_jsons(trainset_dir):
	# TrainSet 디렉토리 내부의 각 세트 폴더 순회
	for set_name in tqdm(os.listdir(trainset_dir), desc='Processing Sets'):
		set_path = os.path.join(trainset_dir, set_name)
		if os.path.isdir(set_path):
			# 세트 폴더 내의 환경(env) 폴더 순회
			for env_folder in os.listdir(set_path):
				env_path = os.path.join(set_path, env_folder)
				cam_down_path = os.path.join(env_path, 'cam_down')

				if os.path.isdir(cam_down_path):
					json_folder = os.path.join(cam_down_path, 'json')
					
					# json 폴더 내의 JSON 파일 순회
					if os.path.exists(json_folder):
						for json_name in tqdm(os.listdir(json_folder), desc=f'Processing {set_name}/{env_folder} JSON files', leave=False):
							json_path = os.path.join(json_folder, json_name)
							if json_path.endswith('.json'):
								try:
									process_json(json_path)
								except Exception as e:
									tqdm.write(f"Error processing {json_name}: {e}")



if __name__ == "__main__":
	# argparse로 경로 인자를 받아옴
	parser = argparse.ArgumentParser(description="Process images in rgba folders.")
	parser.add_argument("trainset_dir", type=str, help="Path to the TrainSet directory")

	args = parser.parse_args()
	
	process_jsons(args.trainset_dir)
	