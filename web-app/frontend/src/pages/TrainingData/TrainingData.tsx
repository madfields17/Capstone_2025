import React from 'react';
import '../shared.css';
import TableauVisualization from './TableauVisualization';

const TrainingData: React.FC = () => {
  return (
    <div>
      <h2>Training Data</h2>
      <h3>Data Balancing</h3>
      <p>Many existing spoofed audio datasets suffer from significant imbalances, often skewed toward specific accents or gender categories. To address this, we curated a training dataset designed to achieve greater balance across both accent and gender, with the goal of supporting more equitable deepfake audio detection research.</p>
      <p>While our dataset currently includes only male and female gender labels due to available metadata, we recognize the importance of broader gender representation and encourage future work in this direction. Similarly, our accent categories are based on broad regional groupings, shaped by practical constraints and existing labeling conventions. We acknowledge that accent classification is a complex and often contested topic, and we hope this work sparks further exploration into more nuanced and inclusive approaches.</p>
      <p>That being said, you can find a visualization of the regional representation of accents in our training set below. Feel free to hover over the visualization for a more granular view of the accent labels. For each of the broader regional categories, we ensured an even split between male and female voices.</p>
      <TableauVisualization/>
      <h3>Data Sources</h3>
      <p>To match the distribution found in existing research, we used a real-to-spoof data split of 37% real data and 63% spoofed data. Once the real data was balanced across accent and gender, we generated the spoof data to mirror this distribution, further minimizing bias.</p>
      <ul style={{
        color: '#33312e',
        margin: '0 auto',
        marginBottom: '15px',
        textAlign: 'justify',
        width: '80%'
      }}>
        <li style={{marginBottom: '15px'}}><strong>Real Data (37% of total data):</strong> The real data was sourced from the Mozilla Common Voice Corpus, ASR Fairness dataset, Singapore National Speech Corpus, Fake-or-Real dataset (only real samples), In-the-Wild dataset (only real samples), and the Speech Accent dataset. To avoid data leakage, different speakers were used for the training and validation sets.</li>
        <li style={{marginBottom: '15px'}}><strong>Spoof Data (63% of total data):</strong> The spoof data was generated using ElevenLab's TTS API. We ensured that different speakers were used for both the training and validation sets, and no speakers were repeated between the real and spoof data.</li>
      </ul>
      <h3>Data Preprocessing</h3>
      <p>After balancing the dataset, we standardized the data by converting all audio files to WAV format, using a mono channel, eliminating noise, trimming silence from the start and end of each clip, setting a sample rate of 16 kHz, keeping the volume at 0 dBFS, and limiting the duration to 5 seconds.</p>
      <h3>Data Partitioning</h3>
      <p>The dataset consists of a total of 8,771 WAV files. These were split into training and validation sets with an 80/20 distribution, meaning the training set contains 6,980 audio files, and the validation set contains 1,791 audio files.</p>
    </div>
  );
};

export default TrainingData;
