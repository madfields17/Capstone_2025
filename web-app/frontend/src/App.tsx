import React, { useState } from 'react';
import Sidebar from './components/Sidebar/Sidebar';
import Home from './pages/Home'; 
import TrainingData from './pages/TrainingData';
import EvaluationData from './pages/EvaluationData';
import AboutModels from './pages/AboutModels';
import DataUsagePrivacy from './pages/DataUsagePrivacy';
import MeetTeam from './pages/MeetTeam';
import './App.css';
import Motivation from './pages/Motivation';

const App: React.FC = () => {
  const [selectedItem, setSelectedItem] = useState<string>('Home'); // Track the selected sidebar item.

  const menuItems = [
    'Home',
    'Motivation',
    'Training Data',
    'Evaluation Data',
    'About the Models',
    'Data Usage and Privacy',
    'Meet the Team',
  ];

  const handleSelect = (item: string) => {
    setSelectedItem(item);
  };

  // Conditionally render the content of the page based on the selected sidebar item.
  const renderPageContent = () => {
    switch (selectedItem) {
      case 'Home':
        return <Home/>;
      case 'Motivation':
        return <Motivation/>
      case 'Training Data':
        return <TrainingData/>;
      case 'Evaluation Data':
        return <EvaluationData/>;
      case 'About the Models':
        return <AboutModels/>;
      case 'Data Usage and Privacy':
        return <DataUsagePrivacy/>;
      case 'Meet the Team':
        return <MeetTeam/>;
      default:
        return <Home/>;
    }
  };

  return (
    <div style={{ display: 'flex' }}>
      <Sidebar menuItems={menuItems} selectedItem={selectedItem} onSelect={handleSelect} />
      <div className='page'>
        {renderPageContent()}
      </div>
    </div>
  );
};

export default App;
