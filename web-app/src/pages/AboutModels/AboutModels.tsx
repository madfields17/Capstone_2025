import '../shared.css';

const AboutModels: React.FC = () => {
  return (
    <div>
      <h2>About the Models</h2>
      <h3>Introduction to Models</h3>
      <p>For our deepfake audio detection, we tested three state-of-the-art model architectures, each chosen for their ability to process audio in an end-to-end fashion without relying on traditional hand-crafted feature engineering. These models represent cutting-edge designs capable of detecting subtle spectral and temporal discrepancies between real and spoofed audio.</p>
      <p>Each of the models we selected leverages different advanced techniques to extract and classify important features from raw audio data:</p>
      <ul style={{
        color: '#33312e',
        margin: '0 auto',
        marginBottom: '15px',
        textAlign: 'justify',
        width: '80%'
      }}>
        <li><strong>AASIST/AASIST-L (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks):</strong> A model utilizing attention mechanisms to focus on the most relevant features in the spectro-temporal domain, providing excellent performance for spoof detection.</li>
        <li><strong>RawNet2</strong>: An end-to-end deep learning model that processes raw waveform data, learning both spectral and temporal patterns directly from the audio, without the need for pre-extracted features.</li>
        <li><strong>Res-TSSDNet (Time-Domain Synthetic Speech Detection Network):</strong> A model designed to detect synthetic speech by analyzing the raw time-domain signal, enabling it to capture subtle artifacts that distinguish real from spoofed audio.</li>
      </ul>
      <p>These models use CNNs, RNNs, and Attention layers and are specifically built to discern intricate patterns and irregularities in audio data, making them highly effective in identifying deepfake audio samples.</p>
      <h3>Model Performance and Fine-Tuning Results</h3>
    </div>
  );
};

export default AboutModels;
