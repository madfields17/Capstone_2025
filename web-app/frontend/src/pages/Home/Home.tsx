import AudioUpload from '../../components/AudioUpload';
import '../shared.css';

const Home: React.FC = () => {
  return (
    <div>
      <h2>Equitable Audio Deepfake Detection</h2>
      <p>Welcome to our platform, where we are developing an innovative, equitable audio deepfake detector that has been carefully balanced across diverse accent categories and genders. In today's digital age, the ability to manipulate audio is rapidly advancing, and it's crucial that detection tools are inclusive, accurate, and fair for everyone, regardless of their accent or gender. Our solution ensures that no group is disproportionately affected by false positives or missed detections, providing a more reliable tool for all. You can test out how our detector works by uploading an audio clip in the section below - give it a try and see the technology in action!</p>
      <AudioUpload/>
    </div>
  );
};

export default Home;
