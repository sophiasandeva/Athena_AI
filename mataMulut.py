# import the necessary packages
import cv2
import dlib
#import time
from scipy.spatial import distance as dist
from imutils import face_utils

#open camera
video = cv2.VideoCapture(0)


# the facial landmark predictor
model = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(model)
# initialize dlib's face detector (HOG-based) and then create
detector = dlib.get_frontal_face_detector()

# fungsi untuk mengenali letak mata dari garis eucladean  
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

# fungsi untuk mengenali letak mulut dari garis eucladean  
def smile(mouth):
    # compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[3], mouth[9]) 
    B = dist.euclidean(mouth[2], mouth[10]) 
    C = dist.euclidean(mouth[4], mouth[8])
    # average form vertical mouth landmark
    L = (A+B+C)/3
    # compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates 
    D = dist.euclidean(mouth[0], mouth[6])
    # compute the mouth aspect ratio  
    mar=(L/D)
    # return the mouth aspect ratio
    return mar
        

# fungsi main / utamanya
def mataMulut():
    while True:
        # read the frame from video
        ret, image = video.read()
        # change the colour of frame to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        faces = detector(gray, 0)
        print("[INFO] loading facial landmark predictor...")
        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        # grab the indexes of the facial landmarks for mouth respectively
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        EYE_AR_THRESH = 0.3
        # konstanta untuk menentukan looping mulut
        TOTAL_MOUTH = 0
        # initialize the total number of blinks
        TOTAL_EYE = 0

        i = 0
        # loop over the face detections
        for face in faces:  # jika ada wajah
            # *** Deteksi Landmark Wajah
            # determine the facial landmarks for the face region
            shape = predictor(gray, face)
            
            # mengincrementkan pengulangan
            i += 1

            #konstanta ntuk mengoperasikan perhitungan koordinat muut dan mata
            pad = 10
            
            # koordinat mata kanan
            # x untuk lebar
            # y untuk tinggi
            xeyeright = [shape.part(x).x for x in range(42, 47)]
            yeyeright = [shape.part(x).y for x in range(42, 47)]
            max_xer = max(xeyeright)
            min_xer = min(xeyeright)
            max_yer = max(yeyeright)
            min_yer = min(yeyeright)
            # letak crop mata kanan
            crop_eyer = image[min_yer - pad : max_yer + pad, min_xer - pad : max_xer + pad]
            
            # koordinat mata kiri
            xeyeleft = [shape.part(x).x for x in range(36, 41)]
            yeyeleft = [shape.part(x).y for x in range(36, 41)]
            max_xel = max(xeyeleft)
            min_xel = min(xeyeleft)
            max_yel = max(yeyeleft)
            min_yel = min(yeyeleft)
            # letak crop mata kiri
            crop_eyel = image[min_yel - pad : max_yel + pad, min_xel - pad : max_xel + pad]
            
            # koordinat mulut
            xmouthpoints = [shape.part(x).x for x in range(48, 67)]
            ymouthpoints = [shape.part(x).y for x in range(48, 67)]
            max_xm = max(xmouthpoints)
            min_xm = min(xmouthpoints)
            max_ym = max(ymouthpoints)
            min_ym = min(ymouthpoints)
            # letak crop mulut
            crop_mouth = image[min_ym - pad : max_ym + pad, min_xm - pad : max_xm + pad]
            
            # convert the facial landmark (x, y)-coordinates to a NumPy
		    # array
            shape = face_utils.shape_to_np(shape)
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            # extract the mouth coordinates, then use the
            # coordinates to compute the mouth aspect ratio for mouth
            mouth = shape[mStart : mEnd]
            mar = smile(mouth)

            #mouthHull = cv2.convexHull(mouth)
            #leftEyeHull = cv2.convexHull(leftEye)
            #rightEyeHull = cv2.convexHull(rightEye)
            #cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(image, [mouthHull], -1, (0, 255, 0), 1)
            
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                TOTAL_EYE = TOTAL_EYE + 1

                #if TOTAL_EYE <= 1: 
                    #time.sleep(0.3)
                    #eyer_name = "eye_right_{}.png".format(TOTAL_EYE)
                    #eyel_name = "eye_left_{}.png".format(TOTAL_EYE)
                    #cv2.imwrite(eyer_name, crop_eyer)
                    #cv2.imwrite(eyel_name, crop_eyel)
                # #print("{} written!".format(ear))
                
                # mengganti mata kanan dan kiri dengan mulut
                image[min_yer - pad : max_yer + pad, min_xer - pad : max_xer + pad] = cv2.resize(crop_mouth, (2 * pad + max_xer - min_xer, 2 * pad + max_yer - min_yer), interpolation = cv2.INTER_AREA)
                image[min_yel - pad : max_yel + pad, min_xel - pad : max_xel + pad] = cv2.resize(crop_mouth, (2 * pad + max_xel - min_xel, 2 * pad + max_yel - min_yel), interpolation = cv2.INTER_AREA)
            
            if mar <= .3 or mar > .38: 
                TOTAL_MOUTH = TOTAL_MOUTH + 1

                #if TOTAL_MOUTH <= 1:
                    #time.sleep(0.3)
                    #img_name = "mouth_{}.png".format(TOTAL_MOUTH)
                    #cv2.imwrite(img_name, crop_mouth)
                #print("{} written!".format(mar))
                
                #mengganti mulut dengan mata
                image[min_ym - pad : max_ym + pad, min_xm - pad : max_xm + pad] = cv2.resize(crop_eyer, (2 * pad + max_xm - min_xm, 2 * pad + max_ym - min_ym), interpolation = cv2.INTER_AREA)

            # nulis angka dari the total number of blinks on the frame along with
            # the computed eye and aspect ratio for the frame
            cv2.putText(image, "Blinks: {}".format(TOTAL_EYE), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # nulis angka dari mult yang tersenyum
            cv2.putText(image, "MAR: {}".format(mar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # menampilkan frame
        cv2.imshow('frame', image)
        # untuk mempertahanakan window agar tetap menampilkan kamera
        k = cv2.waitKey(10)
        # jika menekan tombol q maka framenya nutup
        if k == ord('q'):
            break
        # jika menekan tombol s maka framenya disimpan
        if k == ord('s'):
            cv2.imwrite("mataMulut.png", image)
    # menghentikan operasi streaming dari sisi software maupun hardware.           
    video.release()
    # untuk menutup window lain yang sedang terbuka.
    cv2.destroyAllWindows()

# untuk memanggil fungsi matamulut
if __name__ == '__main__':
    mataMulut()

#image = cv2.imread("static/gambar/Minhoppa.jpeg")
#image = main(image)
#cv2.imshow("Dog Filter",image)
#k = cv2.waitKey()


    
