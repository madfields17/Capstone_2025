import React, { useEffect, useRef } from 'react';
import '../shared.css';

const TableauVisualization: React.FC = () => {
  const vizRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
    script.type = 'text/javascript';
    script.onload = () => {
      const vizElement = vizRef.current?.getElementsByTagName('object')[0];
      if (vizElement) {
        vizElement.style.display = 'block';
        vizElement.style.width = '100%';
        vizElement.style.height = '700px';
      }
    };
    vizRef.current?.appendChild(script);
  }, []);

  return (
    <div className="tableau-container">
      <div
        className="tableauPlaceholder"
        id="viz1744350440187"
        ref={vizRef}
        style={{ position: 'relative' }}
      >
        <noscript>
          <a href="#">
            <img
              alt="FairVoice Eval Dataset Distribution"
              src="https://public.tableau.com/static/images/fa/fairvoiceevaldatasettest4/Sheet1/1_rss.png"
              style={{ border: 'none' }}
            />
          </a>
        </noscript>
        <object className="tableauViz">
          <param name="host_url" value="https%3A%2F%2Fpublic.tableau.com%2F"/>
          <param name="embed_code_version" value="3"/>
          <param
            name="path"
            value="views/fairvoiceevaldatasettest4/Sheet1?:language=en-US&amp;:embed=true&amp;:sid=&amp;:redirect=auth"
          />
          <param name="toolbar" value="yes"/>
          <param
            name="static_image"
            value="https://public.tableau.com/static/images/fa/fairvoiceevaldatasettest4/Sheet1/1.png"
          />
          <param name="animate_transition" value="yes"/>
          <param name="display_static_image" value="yes"/>
          <param name="display_spinner" value="yes"/>
          <param name="display_overlay" value="yes"/>
          <param name="display_count" value="yes"/>
          <param name="language" value="en-US"/>
        </object>
      </div>
    </div>
  );
};

export default TableauVisualization;
