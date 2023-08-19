import { TestBed } from '@angular/core/testing';

import { ObjectDetectionFromYoloToTensorflowService } from './object-detection-from-yolo-to-tensorflow.service';

describe('ObjectDetectionFromYoloToTensorflowService', () => {
  let service: ObjectDetectionFromYoloToTensorflowService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ObjectDetectionFromYoloToTensorflowService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
