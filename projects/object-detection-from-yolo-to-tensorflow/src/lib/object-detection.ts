
import * as tf from '@tensorflow/tfjs'
interface Box {
  boxes: [{
    x: number,
    y: number,
    width: number,
    height: number
  }]
}
export interface Boxes {
  box: {
    x: number,
    y: number,
    width: number,
    height: number
  },
  classeId: number,
  score: number,
}
export default class ObjectDetection {



  protected validate(model: tf.GraphModel<string | tf.io.IOHandler> | null = null, canvas: HTMLCanvasElement | HTMLImageElement | tf.PixelData | ImageData | HTMLVideoElement | ImageBitmap) {
    if (!model) {
      console.error("The model is or invalid or empty/null");
      throw new Error("The model is or invalid or empty/null");
    }
    else if (!canvas) {
      console.error("The canvas is or invalid or empty/null");
      throw new Error("The canvas is or invalid or empty/null");
    }
    else {
      return true;
    }

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
  protected async predictPrivate(model: tf.GraphModel<string | tf.io.IOHandler>, obsToBePredict: HTMLCanvasElement | HTMLImageElement | tf.PixelData | ImageData | HTMLVideoElement | ImageBitmap, threshold: number, imageWidth: number, imageHeight: number): Promise<Boxes[] | null> {
    const ob = new ObjectDetection()

    return await ob.predict(model, obsToBePredict, threshold, imageWidth, imageHeight);
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
  public async predict(model: tf.GraphModel<string | tf.io.IOHandler>, obsToBePredict: HTMLCanvasElement | HTMLImageElement | tf.PixelData | ImageData | HTMLVideoElement | ImageBitmap, threshold: number, imageWidth: number, imageHeight: number): Promise<Boxes[] | null> {

    if (this.validate(model, obsToBePredict)) {


      const conf_threshold = threshold;
      const image = tf.tidy(() => {
        return tf.browser.fromPixels(obsToBePredict)
          .resizeNearestNeighbor([imageWidth, imageHeight])
          .toFloat()
          .expandDims()
          .div(225.0);
      })

      const outputs = model?.predict(image) as tf.Tensor;

      // Step 4: Calcular o valor máximo ao longo do eixo 1 (colunas) usando tf.max
      let { scores, predictions } = await this.getScore(outputs, conf_threshold);


      // Step 10: Obter o número de detecções antes do filtro
      const num_detections_before_filter = scores.shape[0];

      if (num_detections_before_filter > 0) {
        // Step 11: Get the class with the highest confidence

        const classIds = this.getClassIds(predictions)

        const boxes = this.getBoxes(predictions, obsToBePredict, { width: imageWidth, height: imageHeight });

        const indicies = this.nms(boxes, scores, .3);


        if (boxes.shape[0] > 0) {
          const output = this.xywh2xyxy(boxes.arraySync());
          return this.prepareReturn(output, classIds, scores, indicies);
        } else {

          image.dispose();
          outputs.dispose();
          classIds.dispose();
          boxes.dispose();
          tf.dispose();
          tf.disposeVariables();
          return null;
        }
      } else {
        return null;
      }


    } else {
      return null
    }
  }



  /**
   * Get Score
   * @param outputs
   * @param confThreshold
   * @returns Return a promise with score and predictions
   */
  protected async getScore(outputs: tf.Tensor, confThreshold: number): Promise<{ scores: tf.Tensor<tf.Rank>; predictions: tf.Tensor<tf.Rank>; }> {

    let predictions = outputs.squeeze().transpose();
    // Step 2: Fatiar o array 'predictions' para obter apenas as colunas a partir da posição 4

    const sliced_predictions = predictions.slice([0, 4], [-1, -1]);

    // Step 4: Calcular o valor máximo ao longo do eixo 1 (colunas) usando tf.max
    let scores = tf.max(sliced_predictions, 1);

    // Step 6: Criar a máscara booleana para valores acima do limite de confiança
    const above_threshold_mask = tf.greater(scores, confThreshold);

    // Step 7: Filtrar 'predictions' com base na máscara booleana
    predictions = await tf.booleanMaskAsync(predictions, above_threshold_mask);

    // Step 9: Filtrar 'scores' com base na máscara booleana
    scores = await tf.booleanMaskAsync(scores, above_threshold_mask);

    return { scores, predictions }
  }


  /**
   * Get class Ids
   * @param predictions
   * @returns Return ClassIds
   */
  protected getClassIds(predictions: tf.Tensor): tf.Tensor {
    const last4Cols = (predictions.shape[1] || 0) - 3;
    return tf.argMax(predictions.slice([0, last4Cols], [-1, 3]), 1);
  }

  /**
   * Get boxes
   * @param predictions
   * @param canvas
   * @param inputSize  as { width: number, height: number }
   * @returns boxes as Tensor
   */
  protected getBoxes(predictions: tf.Tensor, canvas: HTMLCanvasElement | HTMLImageElement | tf.PixelData | ImageData | HTMLVideoElement | ImageBitmap, inputSize: { width: number, height: number } = { width: 640, height: 480 }): tf.Tensor {
    let boxes = predictions.slice([0, 0], [-1, 4]);
    // Step 12: rescale box

    const input_width = inputSize.width;
    const input_height = inputSize.height;

    const image_width = canvas.width
    const image_height = canvas.height
    const inputShape = tf.tensor1d([input_width, input_height, input_width, input_height], 'float32');
    boxes = boxes.div(inputShape);
    boxes = boxes.mul([image_width, image_height, image_width, image_height]);

    return boxes
  }

  /**
   *
   * @param boxes
   * @param scores
   * @param iou_threshold
   * @returns Indicies
   */
  protected nms(boxes: tf.Tensor, scores: tf.Tensor, iou_threshold: number): number[] {

    const reBoxes = (boxes.arraySync() as any) as number[][]
    const reScore = (scores.arraySync() as any) as number[]
    // Sort by score
    let sortedIndices = this.argSorf(reScore)

    const keep_boxes: number[] = [];
    while (sortedIndices.length > 0) {
      // Pick the last box
      const box_id = sortedIndices[0];
      keep_boxes.push(box_id);

      // Compute IoU of the picked box with the rest
      const box = reBoxes[box_id];
      const index = sortedIndices.length > 1 ? 1 : 0;
      const box2 = sortedIndices.slice(index).map(id => reBoxes[id]);
      const ious = this.compute_iou(box, box2);

      // Remove boxes with IoU over the threshold
      ious.reduce<number[]>((acc, iou, index) => {
        if (iou < iou_threshold) {
          acc.push(index);
        }
        return acc;
      }, []);

      sortedIndices = sortedIndices.slice(1)
    }
    return keep_boxes;
  }

  protected compute_iou(box: number[], boxes: number[][]): number[] {
    // Compute xmin, ymin, xmax, ymax for both boxes

    const col0 = tf.tensor(boxes).slice([0, 0], [-1, 1]).arraySync() as number[]
    const col1 = tf.tensor(boxes).slice([0, 1], [-1, 1]).arraySync() as number[]
    const col2 = tf.tensor(boxes).slice([0, 2], [-1, 1]).arraySync() as number[]
    const col3 = tf.tensor(boxes).slice([0, 3], [-1, 1]).arraySync() as number[]

    const box0 = box[0]
    const box1 = box[1]
    const box2 = box[2]
    const box3 = box[3]

    const xmin = tf.tensor(col0.map(col => Math.max(col, box0)))
    const ymin = tf.tensor(col1.map(col => Math.max(col, box1)))
    const xmax = tf.tensor(col2.map(col => Math.min(col, box2)))
    const ymax = tf.tensor(col3.map(col => Math.min(col, box3)))

    const intersectionArea = tf.maximum(0, xmax.sub(xmin)).mul(tf.maximum(0, ymax.sub(ymin)))

    const boxArea = tf.tensor((box2 - box0) * (box3 - box1))

    const boxesArea = tf.tensor(col2).sub(tf.tensor(col0)).mul(tf.tensor(col3).sub(col1))

    // const union_area = boxes_area.map((area, index) => box_area + area - intersection_area[index]);
    const sumAdd = boxArea.add(boxesArea).transpose();
    const unionArea = sumAdd.sub(intersectionArea);
    const iou = intersectionArea.div(unionArea).dataSync() as any;

    return iou;
  }

  /**
   *
   * @param x
   * @returns
   */
  protected xywh2xyxy(x: any): Box {
    const y = x.map((bbox: any) => {
      return [
        bbox[0] - bbox[2] / 2,
        bbox[1] - bbox[3] / 2,
        bbox[0] + bbox[2] / 2,
        bbox[1] + bbox[3] / 2,
      ];
    }).map((bbox: any) =>

    ({
      x: bbox[0],
      y: bbox[1],
      width: bbox[2],
      height: bbox[3]
    })
    );
    return { boxes: y } as Box;
  }

  /**
   * Arg sorf
   * @param scores
   * @returns Retun the index of scores, from strong score to weak score
   */
  protected argSorf(scores: number[]) {
    return scores.map((value, index) => ({ value, index }))
      .sort((a, b) => a.value - b.value)
      .map(x => x.index)
      .reverse()
  }


  /**
   * Prepare the data return
   * @param data
   * @param classIds
   * @param scores
   * @param indices
   * @returns
   */
  protected prepareReturn(data: Box, classIds: tf.Tensor<tf.Rank>, scores: tf.Tensor<tf.Rank>, indices: number[]): Boxes[] {
    return indices.map(value => {
      return ({
        box: data.boxes[value],
        classeId: (classIds.arraySync() as number[])[value] as number,
        score: (scores.arraySync() as number[])[value] as number * 100,
      })
    })
  }

  /**
   * Get Type Predicted Class
   * @param boxes
   * @param classes
   * @returns Return strings of class
   */
  public getTypePredictedClass(boxes: Boxes[], classes: string[]): string[] {
    let uniques = [] as any[]
    boxes.reduce((acc: any, item) => {
      const { classeId } = item;
      if (!acc[classeId]) {
        acc[classeId] = true;
        uniques.push({ classeId })
      }
      return acc;
    }, {});
    return uniques.map(x => classes[x.classeId]);
  }

  /**
   *
   * @param boxes
   * @param classes
   * @param canvas
   * @param strokeStyle
   * @param lineWidth
   */
  public createBoundingBox(boxes: Boxes[], showBoundingBox: boolean = false, classes: string[], showLegend: boolean = false, canvas: HTMLCanvasElement, strokeStyle: string = 'red', lineWidth: number = 2, verboseOnConsole: boolean = false) {

    const ctx = canvas.getContext('2d')
    let uniques = [] as any[]
    boxes.reduce((acc: any, item) => {
      const { classeId } = item;
      if (!acc[classeId]) {
        acc[classeId] = true;
        uniques.push({ classeId })
      }
      return acc;

    }, {});


    if (ctx != null) {

      uniques.forEach((acc: any) => {

        const id = acc.classeId as number;
        const bestScore = boxes.filter(x => x.classeId === id).reduce((maxScore, item) => {
          return item.score > maxScore ? item.score : maxScore;
        }, -Infinity);
        const boundingBox = boxes.find(x => x.classeId === id && x.score === bestScore);

        if (boundingBox) {
          if (verboseOnConsole) {
            console.log(boundingBox,
              `${classes[boundingBox.classeId]} ${boundingBox.score.toFixed(2)}%`
            )
          }
          ctx.strokeStyle = strokeStyle;
          ctx.lineWidth = lineWidth;
          if (showLegend) {
            ctx.fillText(`${classes[boundingBox.classeId]} ${boundingBox.score.toFixed(2)}%`, boundingBox.box.x, boundingBox.box.y - 5);
          }
          if (showBoundingBox) {

            ctx.strokeRect(
              boundingBox.box.x,
              boundingBox.box.y,
              boundingBox.box.width - boundingBox.box.x,
              boundingBox.box.height - boundingBox.box.y
            );
          }
        }
      })


    }
  }
}
