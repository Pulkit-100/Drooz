import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder


from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread

import numpy as np
import playsound
import argparse
import imutils
import dlib
import random


shape_predictor = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# EYE_AR_THRESH = 0.20
EYE_MIN_CLOSING_TH = 0.1
EYE_CLOSING_TH_RATIO = 1.5

EYE_LR_CONSEC_FRAMES = 60
EYE_TEC_RATIO = 0.50
EYE_TECT_MIN_FRAMES = 500
EYE_BTD_CONSEC_FRAMES = 150
ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()


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


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform, pc):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

        self.pc = pc

        self.channel = pc.createDataChannel("ALARM", negotiated=True, id=0)

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart,
         self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        self.cont_counter = 0  # For Long CLosures (LC)
        self.counter = 0  # For Percent Time with eyes closed (TEC)
        self.btd_counter = 0  # For Blink Total Duration
        self.btd_counter_trig = 0

        self.LC_trig = False
        self.TEC_trig = False
        self.BTD_trig = False

        self.alert_msg = ""

        self.ear = 0

        self.prevAlarmState = False
        self.ALARM_ON = False

        self.total_ear = 0
        self.total_frames = 0
        self.frame_no = 0

        self.open_th = 0.20
        self.closing_th = 0.15

        print("Initialised")

    async def recv(self):
        self.frame_no += 1
        frame = await self.track.recv()
        # print(vars(self.pc))

        if self.transform == "cartoon":
            img = frame.to_ndarray(format="bgr24")

            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "edges":
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "rotate":
            # rotate image
            img = frame.to_ndarray(format="bgr24")
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D(
                (cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        else:
            # print("here")
            # print(type(frame))
            # return frame
            img = frame.to_ndarray(format="bgr24")
            # print(type(img))
            # cv2.imwrite('temp.jpg', img)
            # frame = cv2.imread('temp.jpg')
            # print(type(frame))
            # frame = imutils.resize(frame, width=450)
            # print(type(frame))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print(type(gray))
            # print("HH")
            # # cv2.imwrite('temp2.jpg', img)
            # new_frame = VideoFrame.from_ndarray(gray, format="gray")
            # new_frame.pts = frame.pts
            # new_frame.time_base = frame.time_base
            # print(type(new_frame))
            # return new_frame
            lStart = self.lStart
            lEnd = self.lEnd
            rStart = self.rStart
            rEnd = self.rEnd
            rects = detector(gray, 0)
            # print("Total face detected = ",len(rects))
            # print(rects)
            for rect in rects:
                # print(1)
                shape = predictor(gray, rect)
                # print(2)
                shape = face_utils.shape_to_np(shape)
                # print(3)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                # print(4)
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                # print(5)
                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                # print(ear)
                if ear < self.closing_th:
                    self.total_ear += self.closing_th
                    self.cont_counter += 1
                    # self.counter+=1
                    self.btd_counter_trig = 1
                    self.LC_trig = (self.cont_counter >= EYE_LR_CONSEC_FRAMES)

                    # TEC_trig = self.counter/(self.total_frames+1) > 0.4

                    # if LC_trig or TEC_trig:  # Alert

                    #     if True not in self.ALARM_ON:
                    #         s = ""
                    #         if LC_trig:
                    #             self.ALARM_ON[0]=True
                    #             s="LC Triggered"

                    #         if TEC_trig:
                    #             self.ALARM_ON[1] = True
                    #             s="TEC Triggered"

                    #         cv2.putText(img, s, (170, 25),
                    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    #     # if the alarm is not on, turn it on
                    #     if not self.ALARM_ON:
                    #         self.ALARM_ON = True

                    # cv2.putText(img, "Alert", (180, 50),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # else:
                    #     cv2.putText(frame, "Total face detected = {}".format(len(rects)), (10, 30),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.total_ear += ear
                    self.cont_counter = 0
                    self.LC_trig = False

                TEC_open = True
                if ear >= self.open_th:
                    self.btd_counter = 0
                    self.btd_counter_trig = 0
                    self.TEC_trig = False
                    TEC_open = False

                if ear < (self.open_th+self.closing_th)/2:
                    self.counter += 1

                if self.btd_counter_trig:
                    self.btd_counter += 1

                # cv2.putText(img, "EAR: {:.2f}".format(ear), (300, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, "EAR: {:.2f}".format(ear), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                # self.total_ear += ear
                self.total_frames += 1
                self.open_th = self.total_ear/self.total_frames
                self.closing_th = max(EYE_MIN_CLOSING_TH,
                                      self.open_th/EYE_CLOSING_TH_RATIO)

                # print(self.counter,self.total_frames)
                self.TEC_trig = (self.counter/self.total_frames >
                                 EYE_TEC_RATIO) and TEC_open and self.total_frames > EYE_TECT_MIN_FRAMES

                self.BTD_trig = (self.btd_counter >= EYE_BTD_CONSEC_FRAMES)

                self.ALARM_ON = False

                if self.LC_trig or self.TEC_trig or self.BTD_trig:  # Alert

                    self.ALARM_ON = True

                    if self.LC_trig:
                        self.alert_msg = "LC Triggered"

                    elif self.TEC_trig:
                        self.alert_msg = "TEC Triggered"

                    elif self.BTD_trig:
                        self.alert_msg = "BTD Triggered"

                    cv2.putText(img, self.alert_msg, (130, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    cv2.putText(img, "Alert", (180, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # cv2.putText(img, "Total face detected = {}".format(len(rects)), (10, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                break
            # ret, jpeg = cv2.imencode('.jpg', frame)
            # return jpeg.tobytes()
            cv2.putText(img, "Open Th.: {:.2f}".format(self.open_th), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(img, "Closed Th: {:.2f}".format(self.closing_th), (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            cv2.imwrite('imgs/temp{}.jpg'.format(random.random()), img)

            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            # print(self.total_frames)
            # print(self.cont_counter)
            if self.ALARM_ON != self.prevAlarmState:
                if self.ALARM_ON:
                    self.channel.send("1")
                else:
                    self.channel.send("0")
                self.prevAlarmState = self.ALARM_ON
                # print(self.ALARM_ON)
            return new_frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    # print(offer)
    # print(vars(offer))
    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)
    # print(vars(pc))

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "alarm.wav"))
    if args.write_audio:
        recorder = MediaRecorder(args.write_audio)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)
        # print("pp")

        if track.kind == "audio":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            local_video = VideoTransformTrack(
                track, transform=params["video_transform"], pc=pc
            )
            # print(local_video)
            # print(vars(local_video))

            # print("pp2")
            pc.addTrack(local_video)
            # pc.addTrack(player.audio)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    print("Here")
    application = app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)
