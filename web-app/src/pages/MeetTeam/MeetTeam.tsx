import '../shared.css';
import './MeetTeam.css';
import React from 'react';

const MeetTeam: React.FC = () => {
  const teamMembers = [
    {
      name: 'Alec Naidoo',
      linkedin: 'https://www.linkedin.com/in/alec-naidoo/',
      image: '/images/alec.jpg',  
    },
    {
      name: 'Madison Fields',
      linkedin: 'https://www.linkedin.com/in/madison-fields-401b48139/',
      image: '/images/madison.jpg',
    },
    {
      name: 'Shruti Jain',
      linkedin: 'https://www.linkedin.com/in/shrutijain29/',
      image: '/images/shruti.jpg',
    },
    {
      name: 'Trisha Sanghal',
      linkedin: 'https://www.linkedin.com/in/trishasanghal/',
      image: '/images/trisha.jpg',
    },
  ];

  return (
    <div>
      <h2>Meet the Team</h2>
      <div className='team-grid'>
        {teamMembers.map((member) => (
          <div className='team-member'>
            <img src={member.image} className='team-image'/>
            <div>
              <p className='team-name'>{member.name}</p>
              <p>
                <a href={member.linkedin} target='_blank' rel='noopener noreferrer'>
                  LinkedIn
                </a>
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MeetTeam;
