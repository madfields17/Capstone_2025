import '../shared.css';
import TableauVisualization from './TableauVisualization';

const EvaluationData: React.FC = () => {
  return (
    <div>
      <h2>Evaluation Data</h2>
      <h3>Data Balancing</h3>
      <p>In machine learning, fairness can be defined in several ways, including equality of odds, equality of outcomes, and equality of opportunity. For this project, we've chosen to focus on equality of odds, specifically aiming to achieve equal false positive rates (FPRs) across diverse demographic groups. Our goal is to ensure that individuals from all accents and gender identities are equally likely to be correctly identified as authentic, without being unfairly flagged as deepfakes due to biases inherent in the model.</p>
      <p>To examine FPRs, we curated a balanced dataset that mirrors the accent and gender categories in our evaluation set, using only real audio clips (no spoofed data). The visualization below shows the regional breakdown of accents in our training data, ensuring equal representation from each accent region. We also ensured an equal distribution of male and female voices within each accent region.</p>
      <TableauVisualization/>
      <h3>Data Sources</h3>
      <p>The majority of our data was sourced from Mozilla's Common Voice Corpus, which provided a wide array of authentic speech samples. To ensure a more representative coverage of specific accents and genders, we supplemented this with additional data from other reputable sources.</p>
      <p>For females, we used the Speech Accent Archive to provide samples from three key accent regions: Western Europe and South America, the Middle East, and East Asia. For males, we turned to the ASR Fairness dataset to include more samples from the Middle East and East Asia. This deliberate curation allowed us to maintain balance and diversity within our dataset, ensuring better representation of each accent and gender identity.</p>
      <h3>Data Preprocessing</h3>
      <p>After balancing the dataset, we standardized all audio files to match the preprocessing pipeline used for the training set. This included converting files to WAV format, using a mono channel, eliminating noise, trimming silence from the start and end of each clip, setting a sample rate of 16 kHz, maintaining a volume of 0 dBFS, and limiting the duration to 5 seconds.</p>
    </div>
  );
};

export default EvaluationData;