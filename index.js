import React, { Fragment, useState } from "react";
import ReactDOM from "react-dom";
import { Camera } from "./camera";
import { Button } from "./camera/styles";
import { Root, Preview, Footer, GlobalStyle } from "./styles";

function UserGreeting(propss) {
  return <h1>NORMAL</h1>;
}

function UserGreeting1(propss) {
  return <h1>CRACKED</h1>;
}

function downloadImage(src) {
  const img = new Image();
  img.crossOrigin = 'anonymous';  // This tells the browser to request cross-origin access when trying to download the image data.
  // ref: https://developer.mozilla.org/en-US/docs/Web/HTML/CORS_enabled_image#Implementing_the_save_feature
  img.src = src;
  img.onload = () => {
    // create Canvas
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    // create a tag
    const a = document.createElement('a');
    a.download = 'download.png';
    a.href = canvas.toDataURL('image/png');
    a.click();
  };
}

function App() {
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [cardImage, setCardImage] = useState();
  const [response, setResponse] = useState(null);

  return (
    <Fragment>
      <Root>
        {isCameraOpen && (
          <Camera
            onCapture={blob => setCardImage(blob)}
            onClear={() => setCardImage(undefined)}
          />
        )}

        {cardImage && (
          <div>
            <h2>Preview</h2>
            <Preview src={cardImage && URL.createObjectURL(cardImage)} />
            <div>
            <button onClick={() =>downloadImage( URL.createObjectURL(cardImage))} className="rr">Download</button>
            <style>{"\
        .rr{\
          color:red; position : relative;\
        }\
      "}</style>
            </div>
          </div>
          
        )}
        {!!response && response}

        {cardImage && (
          <button 
          type="submit" 
          value="Add Todo"
          onClick={async () => {
          const response = await fetch("http://127.0.0.1:5000/make-prediction", {
          method: "POST",
          headers: {
          'Content-Type' : 'application/json'
          },
          body: JSON.stringify("'HI':'HELLO'")
          })
          console.log(response);
          if (response.ok) {
            console.log("NORMAL");
            UserGreeting("NORMAL");
            setResponse("NORMAL");
          }
          else {
            console.log("CRACKED");
            UserGreeting1("CRACKED");
            setResponse("Cracked");
          }
        }}
        className="rr">Make Prediction</button>
        )}

        <Footer>
          <button onClick={() => setIsCameraOpen(true)}>Open Camera</button>
          <button
            onClick={() => {
              setIsCameraOpen(false);
              setCardImage(undefined);
            }}
          >
            Close Camera
          </button>
        </Footer>
      </Root>
      <GlobalStyle />
    </Fragment>
  );
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
