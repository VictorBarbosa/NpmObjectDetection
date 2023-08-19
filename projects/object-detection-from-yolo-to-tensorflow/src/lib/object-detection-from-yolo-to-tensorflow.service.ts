import { Injectable } from '@angular/core';
import ObjectDetection, { Boxes } from './object-detection';
import { GraphModel } from '@tensorflow/tfjs';
import { IOHandler } from '@tensorflow/tfjs-core/dist/io/types';

import * as tf from '@tensorflow/tfjs'
import { Observable } from 'rxjs';
@Injectable({
  providedIn: 'root'
})
export class ObjectDetectionFromYoloToTensorflowService extends ObjectDetection {

  constructor() {
    super();
  }

  /**
   *
   * @param model
   * @param obsToBePredict
   * @param classes
   * @param threshold
   * @param imageWidth
   * @param imageHeight
   * @returns Return a promises with a Bounding Box informations and the classe id predicted
   */
  public override async predict(model: GraphModel<string | IOHandler>, obsToBePredict: HTMLCanvasElement | HTMLImageElement | tf.PixelData | ImageData | HTMLVideoElement | ImageBitmap, threshold: number, imageWidth: number, imageHeight: number): Promise<Boxes[] | null> {
    return await this.predictPrivate(model, obsToBePredict, threshold, imageWidth, imageHeight)
  }

  /**
   *
   * @param model
   * @param obsToBePredict
   * @param classes
   * @param threshold
   * @param imageWidth
   * @param imageHeight
   * @returns Return an Observable with a Bounding Box informations and the classe id predicted
   */
  public predictObservable(model: GraphModel<string | IOHandler>, obsToBePredict: HTMLCanvasElement | HTMLImageElement | tf.PixelData | ImageData | HTMLVideoElement | ImageBitmap, threshold: number, imageWidth: number, imageHeight: number): Observable<Boxes[]> {

    return new Observable((ret) => {

      if (this.validate(model, obsToBePredict)) {
        const conf_threshold = threshold;
        const image = tf.tidy(() => {
          return tf.browser.fromPixels(obsToBePredict)
            .resizeNearestNeighbor([imageWidth, imageHeight])
            .toFloat()
            .expandDims()
            .div(225.0);
        });

        const outputs = model?.predict(image) as tf.Tensor;
        // Step 4: Calcular o valor máximo ao longo do eixo 1 (colunas) usando tf.max


        this.getScore(outputs, conf_threshold).then(({ scores, predictions }) => {

          // Step 10: Obter o número de detecções antes do filtro
          const num_detections_before_filter = scores.shape[0];

          if (num_detections_before_filter > 0) {
            // Step 11: Get the class with the highest confidence

            const classIds = this.getClassIds(predictions)

            const boxes = this.getBoxes(predictions, obsToBePredict, { width: imageWidth, height: imageHeight });

            const indicies = this.nms(boxes, scores, .8);

            if (boxes.shape[0] > 0) {
              const output = this.xywh2xyxy(boxes.arraySync());
              const r = this.prepareReturn(output, classIds, scores, indicies);
              ret.next(r)
              ret.complete();
              ret.unsubscribe();

            }

            image.dispose();
            outputs.dispose();
            classIds.dispose();
            boxes.dispose();
            tf.dispose();
            tf.disposeVariables();

          }
        });
      }

    });
  }
}
