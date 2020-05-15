// get audio source
var audio = new Audio("https://fiestoassets.s3.amazonaws.com/Drooz/alarm.wav");
// get DOM elements
var dataChannelLog = document.getElementById("data-channel"),
  iceConnectionLog = document.getElementById("ice-connection-state"),
  iceGatheringLog = document.getElementById("ice-gathering-state"),
  signalingLog = document.getElementById("signaling-state");
console.log(status);

// peer connection
var pc = null;

// data channel
var dc = null,
  dcInterval = null;

function createPeerConnection() {
  var config = {
    sdpSemantics: "unified-plan",
  };

  if (document.getElementById("use-stun").checked) {
    config.iceServers = [
      {
        urls: "stun:stun.l.google.com:19302",
      },
      // },
      // {
      //   'url': 'turn:numb.viagenie.ca:3478?transport=udp',
      //   'username': '100messi.m1@gmail.com',
      //   'credential': 'messi100'
      // },
      // {
      //   'url': 'turn:numb.viagenie.ca:3478?transport=tcp',
      //   'username': '100messi.m1@gmail.com',
      //   'credential': 'messi100'
      // }
    ];
    // config.iceServers = [{urls: ['stun:stun.l.google.com:19302']}];
  }

  pc = new RTCPeerConnection(config);

  // register some listeners to help debugging
  pc.addEventListener("icegatheringstatechange", function () {}, false);
  pc.addEventListener("iceconnectionstatechange", function () {}, false);

  pc.addEventListener("signalingstatechange", function () {}, false);

  // connect audio / video
  pc.addEventListener("track", function (evt) {
    // console.log(evt)
    if (evt.track.kind == "video")
      document.getElementById("video").srcObject = evt.streams[0];
    else document.getElementById("audio").srcObject = evt.streams[0];
  });
  // console.log(pc)
  return pc;
}

function negotiate() {
  return pc
    .createOffer()
    .then(function (offer) {
      return pc.setLocalDescription(offer);
    })
    .then(function () {
      // wait for ICE gathering to complete
      return new Promise(function (resolve) {
        if (pc.iceGatheringState === "complete") {
          resolve();
        } else {
          function checkState() {
            if (pc.iceGatheringState === "complete") {
              pc.removeEventListener("icegatheringstatechange", checkState);
              resolve();
            }
          }
          pc.addEventListener("icegatheringstatechange", checkState);
        }
      });
    })
    .then(function () {
      var offer = pc.localDescription;
      var codec;

      codec = document.getElementById("audio-codec").value;
      if (codec !== "default") {
        offer.sdp = sdpFilterCodec("audio", codec, offer.sdp);
      }

      codec = document.getElementById("video-codec").value;
      if (codec !== "default") {
        offer.sdp = sdpFilterCodec("video", codec, offer.sdp);
      }

      return fetch("/offer", {
        body: JSON.stringify({
          sdp: offer.sdp,
          type: offer.type,
          video_transform: document.getElementById("video-transform").value,
        }),
        headers: {
          "Content-Type": "application/json",
        },
        method: "POST",
      });
    })
    .then(function (response) {
      return response.json();
    })
    .then(function (answer) {
      return pc.setRemoteDescription(answer);
    })
    .catch(function (e) {
      alert(e);
    });
}

function start() {
  pc = createPeerConnection();

  var time_start = null;

  function current_stamp() {
    if (time_start === null) {
      time_start = new Date().getTime();
      return 0;
    } else {
      return new Date().getTime() - time_start;
    }
  }

  if (document.getElementById("use-datachannel").checked) {
    var parameters = JSON.parse(
      document.getElementById("datachannel-parameters").value
    );

    dc = pc.createDataChannel("chat", parameters);
    dc.onclose = function () {
      clearInterval(dcInterval);
    };
    dc.onopen = function () {
      dcInterval = setInterval(function () {
        var message = "ping " + current_stamp();
        dc.send(message);
      }, 1000);
    };
    dc.onmessage = function (evt) {
      if (evt.data.substring(0, 4) === "pong") {
        var elapsed_ms = current_stamp() - parseInt(evt.data.substring(5), 10);
      }
      // console.log(pc)
    };
  }

  var channel = pc.createDataChannel("ALARM", { negotiated: true, id: 0 });
  channel.onmessage = async function (event) {
    let status = await document.getElementById("status-text");

    if (event.data === "1") {
      audio.play();
      status.style.background = "#FF4A4A";
      status.textContent = "Drowsiness Detected";
    } else {
      audio.stop();
      status.style.background = "#a2ffa5";
    }
  };

  var constraints = {
    audio: document.getElementById("use-audio").checked,
    video: false,
  };

  if (document.getElementById("use-video").checked) {
    var resolution = document.getElementById("video-resolution").value;
    if (resolution) {
      resolution = resolution.split("x");
      constraints.video = {
        width: parseInt(resolution[0], 0),
        height: parseInt(resolution[1], 0),
      };
    } else {
      constraints.video = true;
    }
  }

  if (constraints.audio || constraints.video) {
    if (constraints.video) {
    }
    navigator.mediaDevices.getUserMedia(constraints).then(
      function (stream) {
        stream.getTracks().forEach(function (track) {
          pc.addTrack(track, stream);
        });
        return negotiate();
      },
      function (err) {
        alert("Could not acquire media: " + err);
      }
    );
  } else {
    negotiate();
  }
}

const stop = async () => {
  let status = await document.getElementById("status-text");
  console.log(status);

  audio.pause();
  // close data channel
  if (dc) {
    dc.close();
  }

  // close transceivers
  if (pc.getTransceivers) {
    pc.getTransceivers().forEach(function (transceiver) {
      if (transceiver.stop) {
        transceiver.stop();
      }
    });
  }

  // close local audio / video
  pc.getSenders().forEach(function (sender) {
    sender.track.stop();
  });

  // close peer connection
  setTimeout(function () {
    pc.close();
  }, 500);
};

function sdpFilterCodec(kind, codec, realSdp) {
  var allowed = [];
  var rtxRegex = new RegExp("a=fmtp:(\\d+) apt=(\\d+)\r$");
  var codecRegex = new RegExp("a=rtpmap:([0-9]+) " + escapeRegExp(codec));
  var videoRegex = new RegExp("(m=" + kind + " .*?)( ([0-9]+))*\\s*$");

  var lines = realSdp.split("\n");

  var isKind = false;
  for (var i = 0; i < lines.length; i++) {
    if (lines[i].startsWith("m=" + kind + " ")) {
      isKind = true;
    } else if (lines[i].startsWith("m=")) {
      isKind = false;
    }

    if (isKind) {
      var match = lines[i].match(codecRegex);
      if (match) {
        allowed.push(parseInt(match[1]));
      }

      match = lines[i].match(rtxRegex);
      if (match && allowed.includes(parseInt(match[2]))) {
        allowed.push(parseInt(match[1]));
      }
    }
  }

  var skipRegex = "a=(fmtp|rtcp-fb|rtpmap):([0-9]+)";
  var sdp = "";

  isKind = false;
  for (var i = 0; i < lines.length; i++) {
    if (lines[i].startsWith("m=" + kind + " ")) {
      isKind = true;
    } else if (lines[i].startsWith("m=")) {
      isKind = false;
    }

    if (isKind) {
      var skipMatch = lines[i].match(skipRegex);
      if (skipMatch && !allowed.includes(parseInt(skipMatch[2]))) {
        continue;
      } else if (lines[i].match(videoRegex)) {
        sdp += lines[i].replace(videoRegex, "$1 " + allowed.join(" ")) + "\n";
      } else {
        sdp += lines[i] + "\n";
      }
    } else {
      sdp += lines[i] + "\n";
    }
  }

  return sdp;
}

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); // $& means the whole matched string
}
