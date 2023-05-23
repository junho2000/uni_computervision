import cv2
import numpy as np

# Load image pairs - you can change file names
image1 = cv2.imread("/Users/kimjunho/Desktop/컴퓨터비전3-1/[CV]A2/IMG_7577.png") #src
image2 = cv2.imread("/Users/kimjunho/Desktop/컴퓨터비전3-1/[CV]A2/IMG_7578.png") #dst

def get_transform_from_keypoints(img_src, img_dst):
    
    # Create a SIFT object and detect keypoints and descriptors for each image
    sift = cv2.xfeatures2d.SIFT_create() 
    kpts_src, dscrpt_src = sift.detectAndCompute(img_src, None)
    kpts_dst, dscrpt_dst = sift.detectAndCompute(img_dst, None)
    
    # Match the descriptors
    matcher = cv2.FlannBasedMatcher_create()
    matches = matcher.match(dscrpt_src, dscrpt_dst)
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[0:10]

    # Find the homography matrix using the matched keypoints

    src_pts = []
    dst_pts = []
    for match in matches:
        src_pts.append(kpts_src[match.queryIdx].pt)
        dst_pts.append(kpts_dst[match.trainIdx].pt)
    src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    print('src_pts.shape = ', src_pts.shape) #50, 1, 2
    dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
    print('dst_pts.shape = ', dst_pts.shape) #50, 1, 2
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    
    return H, kpts_src, dscrpt_src, kpts_dst, dscrpt_dst, matches

def get_stitched_image(img_src, img_dst, H):

    # Compute the size of the output stitched image
    src_h, src_w = img_src.shape[:2]
    dst_h, dst_w = img_dst.shape[:2]
    src_corners = np.array([[0, 0, 1], [0, src_h, 1], [src_w, src_h, 1], [src_w, 0, 1]], dtype=np.float32)
    warped_corners = np.dot(H, src_corners.T).T
    warped_corners[:, 0] /= warped_corners[:, 2]
    warped_corners[:, 1] /= warped_corners[:, 2]
    min_x = int(np.floor(np.min(warped_corners[:, 0])))
    max_x = int(np.ceil(np.max(warped_corners[:, 0])))
    min_y = int(np.floor(np.min(warped_corners[:, 1])))
    max_y = int(np.ceil(np.max(warped_corners[:, 1])))

    # Modify H to account for the image offset
    stitched_w = max(max_x, dst_w) - min(min_x, 0)
    stitched_h = max(max_y, dst_h) - min(min_y, 0)
    x_offset = min(min_x, 0)
    y_offset = min(min_y, 0)
    H_offset = np.array([[1, 0, -x_offset], [0, 1, -y_offset], [0, 0, 1]], dtype=np.float32)
    H_modified = np.dot(H_offset, H)

    # Warp the first image to the perspective of the second image
    warped_img = cv2.warpPerspective(img_src, H_modified, (stitched_w, stitched_h))
    cv2.imshow('warped_img', warped_img)
    
    # Combine the two images to create a single stitched image
    stitched_image = np.zeros((stitched_h, stitched_w, img_dst.shape[2]), dtype=np.uint8)
    stitched_image[-y_offset:-y_offset+dst_h, -x_offset:-x_offset+dst_w] = img_dst
    stitched_image = cv2.addWeighted(stitched_image, 0.5, warped_img, 0.5, 0.0)

    # return output
    return stitched_image



# 3. Detect and match keypoints and compute transform
H, kpts_src, _, kpts_dst, _, matches = get_transform_from_keypoints(image1, image2)

# 4. Draw the matches on a new image to check validity of matched keypoints
match_image = cv2.drawMatches(image1, kpts_src, image2, kpts_dst, matches, None)
cv2.imshow('matches.png', match_image)

print(H)

# 3. Create stitched image and save
stitched_image = get_stitched_image(image1, image2, H)
cv2.imshow('stitched.png', stitched_image)
cv2.waitKey()
cv2.destroyAllWindows()
    