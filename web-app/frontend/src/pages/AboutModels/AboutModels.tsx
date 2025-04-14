import '../shared.css';

const AboutModels: React.FC = () => {
  return (
    <div>
      <h2>About the Models</h2>
      <h3>Introduction to the Models</h3>
      <p>For our deepfake audio detection, we tested three state-of-the-art model architectures, each chosen for their ability to process audio in an end-to-end fashion without relying on traditional hand-crafted feature engineering. These models represent cutting-edge designs capable of detecting subtle spectral and temporal discrepancies between real and spoofed audio.</p>
      <p>Each of the models we selected leverages different advanced techniques to extract and classify important features from raw audio data:</p>
      <ul style={{
        color: '#33312e',
        margin: '0 auto',
        marginBottom: '15px',
        textAlign: 'justify',
        width: '80%'
      }}>
        <li style={{marginBottom: '15px'}}><strong>AASIST/AASIST-L (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks):</strong> A model utilizing attention mechanisms to focus on the most relevant features in the spectro-temporal domain, providing excellent performance for spoof detection.</li>
        <li style={{marginBottom: '15px'}}><strong>RawNet2</strong>: An end-to-end deep learning model that processes raw waveform data, learning both spectral and temporal patterns directly from the audio, without the need for pre-extracted features.</li>
        <li style={{marginBottom: '15px'}}><strong>Res-TSSDNet (Time-Domain Synthetic Speech Detection Network):</strong> A model designed to detect synthetic speech by analyzing the raw time-domain signal, enabling it to capture subtle artifacts that distinguish real from spoofed audio.</li>
      </ul>
      <p>These models use CNNs, RNNs, and Attention layers and are specifically built to discern intricate patterns and irregularities in audio data, making them highly effective in identifying deepfake audio samples.</p>
      <h3>Model Performance and Fine-Tuning Results</h3>
      <p>For this project, we focused on achieving equality of odds by ensuring equal False Positive Rates (FPRs) across different accents and gender identities. Our goal is to make sure no group is unfairly flagged as a deepfake due to model bias.</p>
      <p>We analyzed FPRs for each demographic group, both before and after fine-tuning our models on a curated training set. To measure fairness, we calculated the Mean Absolute Deviation (MAD) of FPRs, which captures how much each group's FPR deviates from the average. Lower MAD scores indicate more consistent, fairer results across accents and genders.</p>
      <p>Please see our results for each model below.</p>
      <p><strong>AASIST/AASIST-L</strong></p>
      <p>Below, we show the False Positive Rate (FPR) for each accent group before and after fine-tuning the models. As seen in the visualization, the number of false positives decreased significantly. For the AASIST model, the MAD score across accent categories decreased from 0.0839 to 0.0528, indicating a more consistent performance across groups. For the AASIST-L model, the MAD score across accent categories decreased from 0.0737 to 0.0583, indicating a more consistent performance across groups.</p>
      <img src={'/images/aasist-accent.png'} className="fpr-image"/>
      <p>Below, we show the False Positive Rate (FPR) for each gender group before and after fine-tuning the models. As seen in the visualization, the number of false positives decreased significantly. For the AASIST model, the MAD score across gender categories [increased/decreased] from ___ to ___, indicating a [more/less] consistent performance across groups. For the AASIST-L model, the MAD score across gender categories [increased/decreased] from ___ to ___, indicating a [more/less] consistent performance across groups.</p>
      <img src={'/images/aasist-gender.png'} className="fpr-image"/>
      <p><strong>RawNet2</strong></p>
      <p>Below, we show the False Positive Rate (FPR) for each accent group before and after fine-tuning the model. As seen in the visualization, the number of false positives decreased significantly. The MAD score across accent categories decreased from 0.0801 to 0.0379, indicating a more consistent performance across groups.</p>
      <img src={'/images/rawnet2-accent.png'} className="fpr-image"/>
      <p>Below, we show the False Positive Rate (FPR) for each gender group before and after fine-tuning the model. As seen in the visualization, the number of false positives decreased significantly. The MAD score across gender categories increased from 0.014 to 0.015, indicating a less consistent performance across groups.</p>
      <img src={'/images/rawnet2-gender.png'} className="fpr-image"/>
      <p><strong>Res-TSSDNet</strong></p>
      <p>Below, we show the False Positive Rate (FPR) for each accent group before and after fine-tuning the model. As seen in the visualization, the number of false positives decreased significantly. The MAD score across accent categories increased from 0.0348 to 0.1227, indicating a less consistent performance across groups.</p>
      <img src={'/images/res-tssdnet-accent.png'} className="fpr-image"/>
      <p>Below, we show the False Positive Rate (FPR) for each gender group before and after fine-tuning the model. As seen in the visualization, the number of false positives decreased significantly. The MAD score across gender categories decreased from 0.0316 to 0.0038, indicating a more consistent performance across groups.</p>
      <img src={'/images/res-tssdnet-gender.png'} className="fpr-image"/>
      <h3>Best Model</h3>
      <p>While the MAD scores for gender groups didn’t vary dramatically across the models before and after fine-tuning, the differences in performance across accent groups were more pronounced. Among the three, <strong>RawNet2</strong> achieved the lowest Mean Absolute Deviation (MAD) for accent categories, dropping from 0.0801 to 0.0379 after fine-tuning. This indicates RawNet2 was the most consistent and fair across accent groups, which was a key priority for our deployment. For this reason, we selected RawNet2 as the final model used in our demo on the main page.</p>
    </div>
  );
};

export default AboutModels;
