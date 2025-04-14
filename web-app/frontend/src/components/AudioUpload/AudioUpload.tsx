import React, { useRef, useState } from 'react';
import axios from 'axios';

const AudioUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPrediction(null);
      await uploadFile(selectedFile);
    }
  };

  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append('audio', file);
    setIsLoading(true);
    try {
      const response = await axios.post('http://localhost:5050/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setPrediction(response.data?.Prediction ?? 'No prediction found.');
    } catch (error) {
      console.error(error);
      setPrediction('Something went wrong while analyzing the audio.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTryAgain = () => {
    setFile(null);
    setPrediction(null);
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  return (
    <div style={{ width: '80%', margin: '30px auto 0 auto', textAlign: 'left' }}>
      {!file && (
        <>
          <button className="button" onClick={triggerFileInput}>Upload Audio</button>
          <input
            type="file"
            accept="audio/*"
            ref={fileInputRef}
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
        </>
      )}

      {prediction && (
        <div style={{ marginTop: '20px' }}>
          <strong>Model Prediction:</strong>{' '}
          <span style={{ color: prediction.startsWith('Spoofed') ? 'red' : 'green' }}>
            {prediction}
          </span>
          <div>
            <button className="button" onClick={handleTryAgain} style={{ marginTop: '10px' }}>
              Try Again
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioUpload;
