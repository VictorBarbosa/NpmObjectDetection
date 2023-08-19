#### English

This project serves as the foundational base for an NPM library named "object-detection-from-yolo-to-tensorflow". With this library, you can easily take your object detection model created using YOLO and exported to TensorFlow.js. You can generate predictions, create bounding boxes, and determine predicted classes. You can install it using the command `npm install object-detection-from-yolo-to-tensorflow`. The library provides the following methods:

- `predict`: This method returns a Promise.
- `predictObservable`: This method returns an Observable.

Additionally, it includes the following functions:

- `getTypePredictedClass`: This function identifies the predicted class.
- `createBoundingBox`: This function creates a bounding box.

#### Français

Ce projet sert de base fondamentale pour une bibliothèque NPM appelée "object-detection-from-yolo-to-tensorflow". Avec cette bibliothèque, vous pouvez facilement prendre votre modèle de détection d'objets créé à l'aide de YOLO et exporté vers TensorFlow.js. Vous pouvez générer des prédictions, créer des boîtes englobantes et déterminer les classes prédites. Vous pouvez l'installer en utilisant la commande `npm install object-detection-from-yolo-to-tensorflow`. La bibliothèque propose les méthodes suivantes :

- `predict` : Cette méthode renvoie une Promise.
- `predictObservable` : Cette méthode renvoie un Observable.

De plus, elle inclut les fonctions suivantes :

- `getTypePredictedClass` : Cette fonction identifie la classe prédite.
- `createBoundingBox` : Cette fonction crée une boîte englobante.


#### Português

Este projeto serve como base fundamental para uma biblioteca NPM chamada "object-detection-from-yolo-to-tensorflow". Com esta biblioteca, você pode facilmente utilizar o seu modelo de detecção de objetos criado usando YOLO e exportado para TensorFlow.js. Você pode gerar previsões, criar caixas delimitadoras e determinar as classes previstas. Você pode instalá-la usando o comando `npm install object-detection-from-yolo-to-tensorflow`. A biblioteca disponibiliza os seguintes métodos:

- `predict`: Este método retorna uma Promise.
- `predictObservable`: Este método retorna um Observable.

Além disso, ela inclui as seguintes funções:

- `getTypePredictedClass`: Esta função identifica a classe prevista.
- `createBoundingBox`: Esta função cria uma caixa delimitadora.


******************************************************************

```typescript
// Example Usage with observable
this.objectDetect.predictObservable(this.model, canvas, 0.2, 224, 224)
    .pipe(tap(ret => {
        if (ret) {
            const CLASSES = ['orange', 'apple', 'banana'];
            this.objectDetect.createBoundingBox(ret, true, CLASSES, true, canvas, 'blue', 2, true);
            this.objectDetect.getTypePredictedClass(ret, CLASSES).forEach((cls: string) => {
                this.predictText = cls;
            });
        }
    })).subscribe();
 
// Example Usage javascript
this.objectDetect.predict(this.model, canvas, 0.2, 224, 224)
    .then(ret => {
        if (ret) {
            const CLASSES = ['orange', 'apple', 'banana'];
            this.objectDetect.createBoundingBox(ret, true, CLASSES, true, canvas, 'blue', 2, true);
            this.objectDetect.getTypePredictedClass(ret, CLASSES).forEach((cls: string) => {
                this.predictText = cls;
            });
        }
    });
```