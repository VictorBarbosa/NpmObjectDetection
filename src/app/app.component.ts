import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

import { tap } from 'rxjs';
import { ObjectDetectionFromYoloToTensorflowService } from '../../projects/object-detection-from-yolo-to-tensorflow/src/lib/object-detection-from-yolo-to-tensorflow.service';
import { Boxes } from '../../projects/object-detection-from-yolo-to-tensorflow/src/lib/object-detection';
import { positions } from './todelete'
const CLASSES = [
  'Obscene', 'No Obscene'
]
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  title = 'object-detection-from-yolo-to-tensorflow';

  @ViewChild('canvasVideo', { static: true }) canvasVideo: ElementRef<HTMLCanvasElement> | null = null
  @ViewChild('canvasDraw', { static: true }) canvasDraw: ElementRef<HTMLCanvasElement> | null = null
  @ViewChild('canvasDisplay', { static: true }) canvasDisplay: ElementRef<HTMLCanvasElement> | null = null
  @ViewChild('video', { static: true }) video: ElementRef<HTMLVideoElement> | null = null
  @ViewChild('img', { static: true }) img: ElementRef<HTMLImageElement> | null = null
  model: tf.GraphModel<string | tf.io.IOHandler> | null = null;
  predictText: string = ""

  private _firstLine: boolean = false;
  public get firstLine(): boolean {
    return this._firstLine;
  }
  public set firstLine(v: boolean) {
    this._firstLine = v;
  }

  lastMove: any = {};
  /**
   *
   */
  constructor(private objectDetect: ObjectDetectionFromYoloToTensorflowService) {
    // tf.loadGraphModel('/assets/finger_model/model.json')
    // tf.loadGraphModel('/assets/jankenpon_model/model.json')
    tf.loadGraphModel('/assets/obscene_model/model.json')
      .then(model => { this.model = model })

  }

  ngOnInit(): void {
    this.startCam();
  }




  async takePicture() {
    const canvas = this.canvasVideo?.nativeElement;
    const video = this.img?.nativeElement
    const ctx = canvas?.getContext('2d')
    if (canvas && video && ctx) {




      // Wait for the video stream to initialize

      // Set the canvas size equal to the video size
      if (this.canvasVideo && this.video) {

        const videoFrame = () => {

        }

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const ctxDisplay = this.canvasDraw?.nativeElement.getContext('2d');
        if (ctxDisplay) {

          this.predict(ctx, ctxDisplay, canvas);
          // this.joinCanvas();
        }

      }


    }
  }

  async startCam() {
    const canvas = this.canvasVideo?.nativeElement;
    const video = this.video?.nativeElement
    const ctx = canvas?.getContext('2d')
    if (canvas && video && ctx) {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      video.play();

      // Wait for the video stream to initialize
      video.addEventListener('loadedmetadata', () => {
        // Set the canvas size equal to the video size
        if (this.canvasVideo && this.video) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const videoFrame = () => {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          }
          setInterval(() => {
            videoFrame();
            const ctxDisplay = this.canvasDraw?.nativeElement.getContext('2d');
            if (ctxDisplay) {
              this.predict(ctx, ctxDisplay, canvas);
              this.joinCanvas();
            }


          },)
        }
      });

    }
  }

  async predict(ctx: CanvasRenderingContext2D, ctxDisplay: CanvasRenderingContext2D, canvas: HTMLCanvasElement) {
    if (this.model) {


      this.objectDetect.predictObservable(this.model, canvas, 0.2, 224, 224)
        .pipe(tap(ret => {
          if (ret) {

            // console.log(ret)
            this.objectDetect.createBoundingBox(ret, true, CLASSES, true, canvas, 'blue', 2, true);
            this.objectDetect.getTypePredictedClass(ret, CLASSES).forEach((cls: string) => {

              this.predictText = cls;
            })
          }
        }))
        .subscribe()



    }
  }


  /**
   * Draw a circle over the finger
   * @param ctx
   * @param x
   * @param y
   */
  private drawCircle(ctx: CanvasRenderingContext2D, x: number, y: number) {

    const radius = 5;
    const fillColor = 'red';

    ctx.beginPath();
    ctx.arc(x + 20, y + 15, radius, 0, 2 * Math.PI, false);
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.closePath();

    ctx?.beginPath();
  }

  joinCanvas() {

    const canvasVideo = this.canvasVideo?.nativeElement;
    const canvasDraw = this.canvasDraw?.nativeElement; // Substitua 'canvas2' pelo ID do seu segundo canvas
    const canvasDisplay = this.canvasDisplay?.nativeElement;

    if (canvasDraw && canvasVideo && canvasDisplay) {
      const ctxDisplay = canvasDisplay.getContext('2d');

      canvasDisplay.width = canvasVideo.width;
      canvasDisplay.height = canvasVideo.height;

      ctxDisplay?.drawImage(canvasVideo, 0, 0)
      ctxDisplay?.drawImage(canvasDraw, 0, 0)

    }
  }
}
