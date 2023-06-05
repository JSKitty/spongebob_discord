const { InworldClient, InworldPacket, ServiceError, SessionToken, status, Session} = require('@inworld/nodejs-sdk');
const sqlite3 = require('sqlite3').verbose();
const args = process.argv.slice(2);

if (args.length < 7) {
  console.error("Usage: node iw.js [INWORLD_KEY] [INWORLD_SECRET] [INWORLD_SCENE] [text] [user_name] [user_channel] [user_id]");
  process.exit(1);
}
let outputArray = [];
let tempObject = {
  "text": "",
  "audio": ""
};
const INWORLD_KEY = args[0];
const INWORLD_SECRET = args[1];
const INWORLD_SCENE = args[2];
const TEXT = args[3];
const USER_NAME = args[4];
const USER_CHANNEL = args[5];
const USER_ID = args[6];

const getKey = () => `${USER_CHANNEL}_${USER_ID}`;

class Storage {
  constructor(databasePath) {
    this.db = new sqlite3.Database(databasePath);
    this.db.run(`CREATE TABLE IF NOT EXISTS data (key TEXT PRIMARY KEY, value TEXT)`);
  }

  get(key) {
    return new Promise((resolve, reject) => {
      this.db.get('SELECT value FROM data WHERE key = ?', key, (err, row) => {
        if (err) reject(err);
        resolve(row ? JSON.parse(row.value) : null);
      });
    });
  }

  set(key, value) {
    this.db.run('INSERT OR REPLACE INTO data (key, value) VALUES (?, ?)', key, JSON.stringify(value));
  }

  delete(key) {
    this.db.run('DELETE FROM data WHERE key = ?', key);
  }
}

const storage = new Storage('db/data.db');

async function do_inworld_chat(text, user_name) {
  let messages = []; 
  const key = getKey();
  const inworldClient = new InworldClient()
    .setOnSession({
      get: async () => storage.get(key),
      set: (session) => storage.set(key, session),
    })
    .setApiKey({
      key: INWORLD_KEY,
      secret: INWORLD_SECRET,
     })
    .setUser({ fullName: user_name })
    .setConfiguration({
      capabilities: { audio: true, emotions: false },
    })
    .setScene(INWORLD_SCENE)
    .setOnError(async (err) => {
      if (err.message !== '1 CANCELLED: Cancelled on client') {
        console.error(err);
        await storage.delete(key);
        await do_inworld_chat(text, user_name, storage);
      }
    })
    .setOnMessage((packet) => {

      if (packet.isInteractionEnd()) {
          const jsonString = JSON.stringify(outputArray);
          console.log(jsonString);
          connection.close();
      } else if (packet.isText()) {
        tempObject.text = packet.text.text;
        outputArray.push(tempObject);
      }
})
  const connection = inworldClient.build();
  await connection.sendText(text);
}

do_inworld_chat(TEXT, USER_NAME).catch();
