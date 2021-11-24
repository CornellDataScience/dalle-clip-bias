module.exports = {
    purge: ['./src/**/*.{js,jsx,ts,tsx}', './public/index.html'],
    darkMode: false, // or 'media' or 'class'
    theme: {
      extend: {
        outline: {
          grey: '2px solid #808080'
        }
      },
    },
    variants: {
      extend: {},
    },
    plugins: [],
  }