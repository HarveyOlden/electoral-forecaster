// Site settings: the only file you need to touch for contact details, data location, refresh timing and the mascot.
export const CONFIG = {
  // Where the model's output lives. "data/" = files in this site's own repo (what the GitHub job updates).
  // To serve data from elsewhere later, put the full URL here (e.g. "https://raw.githubusercontent.com/<you>/<repo>/main/data/").
  dataBase: "data/",

  contact: {
    email: "your.name@gmail.com",                          // <- change
    linkedin: "https://www.linkedin.com/in/your-profile/",  // <- change
  },

  // How often an open page re-checks for new model output (minutes). The live tab polls faster.
  refreshMinutes: { forecast: 15, live: 1 },

  // Tabs, in order. To add a tab: make js/tabs/<id>.js exporting render(el, ctx), then add a line here.
  tabs: [
    { id: "forecast", label: "Forecast" },
    { id: "live", label: "Live results", live: true },
    { id: "track", label: "Track record" },
    { id: "how", label: "How it works" },
    { id: "sources", label: "Data sources" },
  ],

  mascot: {
    idleFrames: 8,               // assets/mascot/idle_1.png ... idle_8.png (missing frames are skipped)
    idleFps: 1.5,
    anchorFacesRight: true,      // mirror the desk dragon so he faces (and points at) the numbers
    peekStates: { normal: "peek_normal.png", winkLeft: "peek_wink_left.png", winkRight: "peek_wink_right.png" },
    winkEverySeconds: [2.5, 7],  // random gap between winks
  },

  // Probability bands for the map colours (chance the Democrat wins).
  ratings: [
    { min: 0.95, label: "Safe Democratic", key: "d-safe" },
    { min: 0.8, label: "Likely Democratic", key: "d-likely" },
    { min: 0.6, label: "Leans Democratic", key: "d-lean" },
    { min: 0.4, label: "Toss-up", key: "tossup" },
    { min: 0.2, label: "Leans Republican", key: "r-lean" },
    { min: 0.05, label: "Likely Republican", key: "r-likely" },
    { min: -1, label: "Safe Republican", key: "r-safe" },
  ],
};
