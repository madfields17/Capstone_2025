import './Sidebar.css';

const Sidebar = ({ menuItems, selectedItem, onSelect }: any) => {
  return (
    <div className="sidebar">
      <h2>Menu</h2>
      <ul>
        {menuItems.map((item: String) => (
          <li>
            <a
              href="#"
              className={selectedItem === item ? 'selected' : ''}
              onClick={(e) => {
                e.preventDefault(); // Prevent default anchor link behavior
                onSelect(item); // Set selected item
              }}
            >
              {item}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Sidebar;
