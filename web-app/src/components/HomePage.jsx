import React, { Component } from 'react';
import '../App.css';

export default class Homepage extends Component {
    constructor(props) {
        super(props);
        this.state = {
            currentValue: 0,
            currentImage: 'output.png',
            imagePrompt: 'Prompt Text',
            outputImage: 'output.png'
        }
    }

    updateImage( imagePrompt, outputImage ) {
        fetch('/clip?prompt=' + imagePrompt + '&outputImage=' + outputImage).then(res => res.json()).then(data => {
                console.log(data.output);
                console.log('NEW STATE');
                console.log(this.state);
                this.setState({
                    currentImage: data.output,
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
                                <div class="outline-grey mx-96 px-16">
                                    <h1 class="text-blue-700 text-2xl font-extrabold text-center pt-2">Image Generation: CLIP</h1>
                                    {/* <p class="tracking-widest">{currentValue}</p> */}
                                    <label class = "px-8" for="fname">Image Prompt:</label>
                                    <input class = "py-2 w-32" type = "text" value = {imagePrompt} onChange = {(event) => this.updateImagePrompt(event)}/>
                                    <button class = "outline-grey mx-20 px-10" onClick = {() => this.updateImage(imagePrompt, outputImage)
                                        }>Run CLIP</button>
                                    <img class = "py-4 mx-7" src = {currentImage} alt = 'Current Output' />
                                    {/* <img class = "py-4 mx-20" src = 'output.png' /> */}
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

