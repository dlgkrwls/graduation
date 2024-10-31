var streamVideo
if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia )
{
    alert("Media Device not supported")
} else {
    document.getElementById("openBtn").addEventListener('click',open)
    document.getElementById("closeBtn").addEventListener('click',close)
    open()
}
function open() {
    close()
    navigator.mediaDevices.getUserMedia({video:true}).then(stream => {
    streamVideo = stream
    var cameraView = document.getElementById("cameraview");
    cameraView.srcObject = stream;
    cameraView.play()
    })
}
function close() {
    if (streamVideo) {
    var track = streamVideo.getTracks()
    track[0].stop()
    streamVideo = null
    }
}
