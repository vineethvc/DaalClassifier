

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const tfn = require('@tensorflow/tfjs-node');
import {node} from "@tensorflow/tfjs-node";

const cocossd = require('@tensorflow-models/coco-ssd');
const mobilenet = require('@tensorflow-models/mobilenet');

import toUint8Array from 'base64-to-uint8array';


export default class ObjectDetectors {

    constructor(image, type) {

        this.inputImage = image;
        this.type = type;
    }
    
    async loadCocoSsdModal() {
        const modal = await cocossd.load({
            base: 'mobilenet_v2'
        })
        return modal;
    }

    async loadMobileNetModal() {
        const modal = await mobilenet.load({
            version: 1,
            alpha: 0.25 | .50 | .75 | 1.0,
        })
        return modal;
    }

    async loadSavedModel() {
        const handler = tfn.io.fileSystem("./static/savedModel/model.json");
        //const model = await tf.loadLayersModel("http://localhost:3000/savedModel/model.json");
        const model = await tf.loadLayersModel(handler);
        console.log("model loaded");
        return model;
    }

    getTensor3dObject(numOfChannels) {

        const imageData = this.inputImage.replace('data:image/jpeg;base64','')
                            .replace('data:image/png;base64','');
        
        const imageArray = toUint8Array(imageData);
        //const tensor3d = node.decodeJpeg( imageArray, numOfChannels );

        let tensor3d = node.decodeJpeg( imageArray, numOfChannels ).resizeNearestNeighbor([148, 148]).toFloat().expandDims();
        //tensor3d = tensor3d.reshape();
        console.log("Testing:  "+tensor3d.shape)
        return tensor3d;
    }

    async process() {
          
        let predictions = null;
        const tensor3D = this.getTensor3dObject(3);
        let className = null;

        if(this.type === "imagenet") {
            // const model =  await this.loadMobileNetModal();
            // predictions = await model.classify(tensor3D);
            const model =  await this.loadSavedModel();
            let resizeTensor = tf.tensor1d([255]);
            let processedTensor = tensor3D.div(resizeTensor)
            predictions = await model.predict(processedTensor);
            let probability = predictions.arraySync()[0,0][0];
            if(probability > 0.5){
                predictions = [{className: 'Urad Daal', probability: probability}];
            } else {
                console.log(probability);
                probability = (1 - probability)*100;
                predictions = [{className: 'Toor Daal', probability: probability}];
            }
            console.log(predictions);
        } else {

            const model =  await this.loadCocoSsdModal();
            predictions = await model.detect(tensor3D);
        }

        tensor3D.dispose();
       return {data: predictions, type: this.type};
    }
}
