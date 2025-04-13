import '../shared.css';

const DataUsagePrivacy: React.FC = () => {
  return (
    <div>
      <h2>Data Usage and Privacy</h2>
      <p>
        We are dedicated to ensuring the privacy and security of your data. By submitting an audio clip, you agree to the following terms regarding its collection, storage, and use:
      </p>
      <p>
        1. <strong>Data Collection and Storage:</strong> We are committed to protecting your privacy and ensuring the security of your data. By submitting an audio clip, you acknowledge and agree that the clip will be temporarily stored in a secure Amazon S3 bucket for the sole purpose of processing and analysis by our model. Please refrain from including any personally identifiable information (PII) in your submission, as we cannot be held responsible for any data that may inadvertently contain such information.
      </p>
      <p>
        2. <strong>Data Usage:</strong> The audio file will only be used to conduct analysis via our model. It will not be used for any other purpose. Once the analysis is completed, the audio file will be automatically deleted from our systems.
      </p>
      <p>
        3. <strong>Data Retention:</strong> The audio file will be retained solely for the duration of the processing and inference. We do not store any data longer than necessary to complete the analysis process. After the process is complete, the file will be deleted.
      </p>
      <p>
        4. <strong>Data Security:</strong> We take the security of your data seriously. All measures, including secure storage and access controls, are in place to protect your audio submission from unauthorized access. We do not share, sell, or distribute your audio clip to any third party.
      </p>
      <p>
        5. <strong>Your Privacy:</strong> Your privacy is our top priority. We are committed to safeguarding your data and complying with applicable privacy laws and regulations.
      </p>
      <p>
        By submitting your audio clip, you acknowledge that you understand and agree to these terms.
      </p>
    </div>
  );
};

export default DataUsagePrivacy;
