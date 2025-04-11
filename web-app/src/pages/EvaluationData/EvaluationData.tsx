import '../shared.css';
import TableauVisualization from './TableauVisualization';

const EvaluationData: React.FC = () => {
  return (
    <div>
      <h2>Evaluation Data</h2>
      <h3>Data Balancing</h3>
      <p>Info Here</p>
      <TableauVisualization/>
      <h3>Data Sources</h3>
      <p>Info Here</p>
      <h3>Data Preprocessing</h3>
      <p>After balancing the dataset, we standardized all audio files to match the preprocessing pipeline used for the training set. This included converting files to WAV format, using a mono channel, eliminating noise, trimming silence from the start and end of each clip, setting a sample rate of 16 kHz, maintaining a volume of 0 dBFS, and limiting the duration to 5 seconds.</p>
    </div>
  );
};

export default EvaluationData;