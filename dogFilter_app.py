import cv2
import cv2
import math
import dlib
from imutils import face_utils, rotate_bound

# Menggambar sprite di atas gambar
# Ini menggunakan saluran alfa untuk melihat piksel mana yang perlu diganti
# Input: gambar, sprite: numpy arrays
# output: gambar gabungan yang dihasilkan
def draw_sprite(frame, sprite, x_offset, y_offset):
    (h, w) = (sprite.shape[0], sprite.shape[1])
    (imgH, imgW) = (frame.shape[0], frame.shape[1])

    if y_offset + h >= imgH:  # jika sprite keluar dari gambar di bawah
        sprite = sprite[0 : imgH - y_offset, :, :]

    if x_offset + w >= imgW:  # jika sprite keluar dari gambar ke kanan
        sprite = sprite[:, 0 : imgW - x_offset, :]

    if x_offset < 0:  # jika sprite keluar dari gambar ke kiri
        sprite = sprite[:, abs(x_offset) : :, :]
        w = sprite.shape[1]
        x_offset = 0

    # Untuk setiap RGB chanel
    for c in range(3):
        # chanel 4 adalah alpha: 255 adalah tidak transparan, 0 adalah background transparan
        frame[y_offset : y_offset + h, x_offset : x_offset + w, c] = sprite[:, :, c] * (
            sprite[:, :, 3] / 255.0
        ) + frame[y_offset : y_offset + h, x_offset : x_offset + w, c] * (
            1.0 - sprite[:, :, 3] / 255.0
        )
    return frame

# Sesuaikan sprite yang diberikan dengan lebar dan posisi kepala
# jika sprite tidak pas dengan layar di bagian atas, sprite harus dipangkas
def adjust_sprite2head(sprite, head_width, head_ypos, ontop=True):
    (h_sprite, w_sprite) = (sprite.shape[0], sprite.shape[1])
    factor = 1.0 * head_width / w_sprite
    sprite = cv2.resize(
        sprite, (0, 0), fx=factor, fy=factor
    )  # sesuaikan agar memiliki lebar yang sama dengan kepala
    (h_sprite, w_sprite) = (sprite.shape[0], sprite.shape[1])

    y_orig = (
        head_ypos - h_sprite if ontop else head_ypos
    )  # sesuaikan posisi sprite ke ujung di mana kepala dimulai
    if (
        y_orig < 0
    ):  # periksa apakah kepala tidak dekat dengan bagian atas gambar dan sprite tidak muat di layar
        sprite = sprite[abs(y_orig) : :, :, :]  # dalam hal ini, kami memotong sprite
        y_orig = 0  # sprite kemudian dimulai di bagian atas gambar
    return (sprite, y_orig)

# Menerapkan sprite ke gambar koordinat wajah yang terdeteksi dan menyesuaikannya dengan kepala
def apply_sprite(image, path2sprite, w, x, y, angle, ontop=True):
    sprite = cv2.imread(path2sprite, -1)
    # print sprite.shape
    sprite = rotate_bound(sprite, angle)
    (sprite, y_final) = adjust_sprite2head(sprite, w, y, ontop)
    image = draw_sprite(image, sprite, x, y_final)

# titik adalah tupel dalam bentuk (x,y)
# mengembalikan sudut antara titik dalam derajat
def calculate_inclination(point1, point2):
    x1, x2, y1, y2 = point1[0], point2[0], point1[1], point2[1]
    incl = 180 / math.pi * math.atan((float(y2 - y1)) / (x2 - x1))
    return incl

# Mengisi variabel dengan koordinat
def calculate_boundbox(list_coordinates):
    x = min(list_coordinates[:, 0])
    y = min(list_coordinates[:, 1])
    w = max(list_coordinates[:, 0]) - x
    h = max(list_coordinates[:, 1]) - y
    return (x, y, w, h)

# Menghubungkan anggota badan dari muka degan koordinatnya
def get_face_boundbox(points, face_part):
#    if face_part == 1:
#        (x, y, w, h) = calculate_boundbox(points[17:22])  # alis kiri
#    elif face_part == 2:
#        (x, y, w, h) = calculate_boundbox(points[22:27])  # alis kanan
#    elif face_part == 3:
#        (x, y, w, h) = calculate_boundbox(points[36:42])  # mata kiri
#    elif face_part == 4:
#        (x, y, w, h) = calculate_boundbox(points[42:48])  # mata kanan
    if face_part == 5:
        (x, y, w, h) = calculate_boundbox(points[29:36])  # hidung
    elif face_part == 6:
        (x, y, w, h) = calculate_boundbox(points[48:68])  # mulut
    return (x, y, w, h)

class VideoCameraSatu(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        ret, image = self.video.read()
        
        # Landmark wajah
        print("[INFO] loading facial landmark predictor...")
        model = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(model)

        # Jalur filter
        detector = dlib.get_frontal_face_detector()
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:  # jika ada wajah
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            # *** Deteksi Landmark Wajah
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            incl = calculate_inclination(
                shape[17], shape[26]
            )  # kemiringan berdasarkan alis

            # kondisi untuk melihat apakah mulut terbuka
            is_mouth_open = (
                shape[66][1] - shape[62][1]
            ) >= 10  # y koordinat titik bibir landmark
            (x0, y0, w0, h0) = get_face_boundbox(shape, 6)  # kotak mulut terikat

            (x3, y3, w3, h3) = get_face_boundbox(shape, 5)  # hidung
            apply_sprite(image, "static/gambar/dogs_nose.png", w3, x3, y3, incl, ontop=False)
            apply_sprite(image, "static/gambar/dog_ears.png", w, x, y, incl) #telinga

            if is_mouth_open:
                apply_sprite(image, "static/gambar/dogs_tongue.png", w0, x0, y0, incl, ontop=False) #lidah

        # OpenCV mewakili gambar sebagai BGR; PIL tapi RGB, kita perlu mengubah urutan chanel
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()