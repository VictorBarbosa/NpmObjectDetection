import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppComponent } from './app.component';
import { ObjectDetectionFromYoloToTensorflowModule } from '../../projects/object-detection-from-yolo-to-tensorflow/src/lib/object-detection-from-yolo-to-tensorflow.module';


@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    BrowserModule
  ],
  providers: [ObjectDetectionFromYoloToTensorflowModule],
  bootstrap: [AppComponent]
})
export class AppModule { }
