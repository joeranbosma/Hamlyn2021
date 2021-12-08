import os
import argparse

import cv2
import open3d as o3d

from slam import process
from display import Display
from pointmap import PointMap

pmap = PointMap()
display = Display()

def main():
	cap = cv2.VideoCapture("videos/scene_4_translation_sequence_5hz.mp4")

	pcd = o3d.geometry.PointCloud()
	visualizer = o3d.visualization.Visualizer()
	visualizer.create_window(window_name="3D plot", width=960, height=540)

	while cap.isOpened():
		ret, frame = cap.read()
		try:
			frame = cv2.resize(frame, (960, 540))
			img, tripoints, kpts, matches = process(frame)
			xyz = pmap.collect_points(tripoints)
		except:
			break

		if ret:
			if kpts is not None or matches is not None:
				display.display_points2d(frame, kpts, matches)
			else:
				pass
			display.display_vid(frame)

			if xyz is not None:
				display.display_points3d(xyz, pcd, visualizer)
			else:
				pass
			if cv2.waitKey(1) == 27:
				break
		else:
			break

	cv2.destroyAllWindows()
	cap.release()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='HamlynUNet')

	## ARGS
	parser.add_argument("--path_data",
                        "-d",
                        help="Path to data for simulation",
                        required=True,
                        type=str,
                        default=None)

	main()

	args = parser.parse_args()

	# depth map and 3D coords reading test
	depth = cv2.imread(os.path.join(args.path_data, "depth_sequences/sequences/scene_4/depth/depth0099.exr"),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
	print("depth = {}".format(depth))
	coords = cv2.imread(os.path.join(args.path_data, "3Dcoordinates_sequences/sequences/scene_4/3Dcoordinates/coords0099.exr"),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
	print("3D coords = {}".format(coords))
