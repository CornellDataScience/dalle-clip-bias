import React, { Component } from 'react';
import '../App.css';

export default class Homepage2 extends Component {
    constructor(props) {
        super(props);
        this.state = {
            currentValue: 0,
            currentImage: '',
            imagePrompt: '',
            outputImage: './output.png'
        }
    }

    updateImage( imagePrompt, outputImage ) {
        fetch('/clip?prompt=' + imagePrompt + '?outputImage=' + outputImage).then(res => res.json()).then(data => {
                console.log(data.output);
                this.setState({
                    currentImage: data.output
                });
        })
    }

    updateImagePrompt( event ) {
        this.setState({
          imagePrompt: event.target.value
        });
      }

    render() {
        console.log("RENDER");
        console.log(this.state);
        let { currentValue, currentImage, imagePrompt, outputImage } = this.state;
        if (this.props.display === 0) {
            return (
                <div className="HomePage">
                    <header class="bg-white shadow">
                        <div class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
                            <h1 class="text-3xl font-bold text-gray-900">Home Page</h1>
                        </div>
                    </header>
                    <main>
                        <div class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                            <div class="px-4 py-6 sm:px-0">
                                <div class="border-4 border-dashed border-gray-200 rounded-lg h-96">
                                    <h1 class="text-blue-400 font-extrabold">Test2</h1>
                                    <p class="tracking-widest">{currentValue}</p>
                                    <label for="fname">Image Prompt:</label>
                                    <input class="py-2" type = "text" value = {imagePrompt} onChange = {(event) => this.updateImagePrompt(event)}/>
                                    <br />
                                    <button onClick = {() => this.updateImage(imagePrompt, outputImage)
                                    }>Run CLIP</button>
                                    <img src = {currentImage} alt = 'Current CLIP Output' />
                                </div>
                            </div>
                        </div>
                    </main>
                </div>
            )
        } else {
            return <div className="HomePage"></div>
        }
    }
}

