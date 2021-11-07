import React, { Component } from 'react';
import Navbar from './Navbar';
import HomePage from './HomePage';
import '../App.css';

export default class WebApp extends Component {
    constructor(props) {
        super(props);
        this.state =  {
            page: 0
        }
    }

    render() {
        return (
            <div className="WebApp">
                <Navbar toCallBack={(childState) => this.setState({page: childState.page})}/>
                <HomePage display={this.state.page} />
            </div>
        )
    }
}