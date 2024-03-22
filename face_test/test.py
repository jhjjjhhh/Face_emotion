import cv2
import dlib
import numpy as np
#from keras.models import load_model
from tensorflow.keras.models import load_model



# 얼굴 인식을 위한 Haar Cascade 분류기
FACE_CASCADE_PATH = '/home/ubu/Downloads/model/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# 얼굴 특징 추출을 위한 얼굴 랜드마크 예측기
PREDICTOR_PATH = '/home/ubu/Downloads/model/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# 표정 라벨링
EXPRESSION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 표정 가중치 모델
MODEL_PATH = '/home/ubu/Downloads/model/emotion_model.hdf5'
model = load_model(MODEL_PATH)


# 비디오 실행
video_capture = cv2.VideoCapture(0)

while True:
    # ret, frame 반환
    ret, frame = video_capture.read()

    if not ret:
        break

    # 얼굴인식을 위해 gray 변환
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 인식
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 얼굴이 인식되면 얼굴 랜드마크 추출
    for (x, y, w, h) in faces:
        # 얼굴 영역에 초록색 사각형 그리기
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 얼굴 영역 추출
        face_rect = dlib.rectangle(x, y, x+w, y+h)

        # 얼굴 랜드마크 예측
        landmarks = predictor(gray, face_rect)

        # 얼굴 특징점 좌표를 리스트로 저장
        landmarks_points = []
        for n in range(0, 68):
            x_landmark = landmarks.part(n).x
            y_landmark = landmarks.part(n).y
            landmarks_points.append((x_landmark, y_landmark))

        # 얼굴 특징점 좌표로 표정을 인식하기 위한 작업 수행
        # 표정 분석을 위해 얼굴 영역을 64x64 크기로 조정
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0

        # 모델을 통해 표정 분석
        output = model.predict(face_roi)[0]

        # 해당 표정의 값 반환
        expression_index = np.argmax(output)

        # 표정에 따른 label 값 저장
        expression_label = EXPRESSION_LABELS[expression_index]

        # 표정 값 출력
        cv2.putText(frame, expression_label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    # 출력
    cv2.imshow('Expression Recognition', frame)

    # esc 누를 경우 종료
    key = cv2.waitKey(25)
    if key == 27:
        break

if video_capture.isOpened():
    video_capture.release()
cv2.destroyAllWindows()
