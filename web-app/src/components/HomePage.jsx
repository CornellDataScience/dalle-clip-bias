import React, { useState, useEffect } from 'react';
import '../App.css';

function HomePage(props) {
    const [currentValue, updateValue] = useState('test-start');
    const [currentImage, updateImage] = useState('');

    useEffect(() => {
        // fetch('/test').then(res => res.json()).then(data => {
        //     updateValue(data.output);
        // })
        //fetch('/clip').then(res => res.json()).then(data => {
        //    console.log(data.output);
        //})
    }, [currentImage]);

    if (props.display === 0) {
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
                                <button onClick={() => updateImage('./output.png')
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

export default HomePage;